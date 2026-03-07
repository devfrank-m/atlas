use std::fs::{self, File};
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use super::VectorBlobRef;
use super::vector_storage::VectorStorage;

// magic(4) + version(4) + dimension(4) + count(8)
pub const HEADER_SIZE: usize = 20;

const MAGIC: &[u8; 4] = b"ATVX";
const VERSION: u32 = 1;

pub fn vec_path(data_dir: &Path, id: &str) -> PathBuf {
    data_dir.join(format!("{id}.vec"))
}

pub fn write(
    data_dir: &Path,
    id: &str,
    vectors: &[f32],
    dimension: u32,
) -> Result<VectorBlobRef, String> {
    let count = if dimension == 0 {
        0
    } else {
        vectors.len() as u64 / dimension as u64
    };
    let path = vec_path(data_dir, id);
    let tmp = path.with_extension("vec.tmp");

    let mut buf = Vec::with_capacity(HEADER_SIZE + vectors.len() * 4);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&dimension.to_le_bytes());
    buf.extend_from_slice(&count.to_le_bytes());
    buf.extend_from_slice(bytemuck::cast_slice(vectors));

    fs::write(&tmp, &buf).map_err(|e| e.to_string())?;
    fs::rename(&tmp, &path).map_err(|e| e.to_string())?;

    Ok(VectorBlobRef {
        file: format!("{id}.vec"),
        count,
        dimension,
        offset_bytes: HEADER_SIZE as u64,
    })
}

pub fn load_mmap(data_dir: &Path, blob: &VectorBlobRef) -> Result<VectorStorage, String> {
    let path = data_dir.join(&blob.file);
    let file = File::open(&path).map_err(|e| format!("{}: {e}", blob.file))?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;

    validate_header(&mmap, blob)?;

    Ok(VectorStorage::Mapped(mmap))
}

fn validate_header(mmap: &Mmap, blob: &VectorBlobRef) -> Result<(), String> {
    if mmap.len() < HEADER_SIZE {
        return Err("vec file too small".into());
    }
    if &mmap[..4] != MAGIC {
        return Err("vec file has invalid magic".into());
    }
    let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(format!("unsupported vec version: {version}"));
    }
    let dimension = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
    if dimension != blob.dimension {
        return Err(format!(
            "dimension mismatch: expected {}, got {dimension}",
            blob.dimension
        ));
    }
    let count = u64::from_le_bytes(mmap[12..20].try_into().unwrap());
    if count != blob.count {
        return Err(format!(
            "count mismatch: expected {}, got {count}",
            blob.count
        ));
    }
    Ok(())
}
