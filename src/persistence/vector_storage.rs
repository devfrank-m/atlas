use memmap2::Mmap;

use super::vectors::HEADER_SIZE;

pub enum VectorStorage {
    Owned(Vec<f32>),
    Mapped(Mmap),
}

impl VectorStorage {
    pub fn as_slice(&self) -> &[f32] {
        match self {
            VectorStorage::Owned(v) => v.as_slice(),
            VectorStorage::Mapped(mmap) => bytemuck::cast_slice(&mmap[HEADER_SIZE..]),
        }
    }

    pub fn ensure_owned(&mut self) {
        if let VectorStorage::Mapped(_) = self {
            let data = self.as_slice().to_vec();
            *self = VectorStorage::Owned(data);
        }
    }

    pub fn as_owned_mut(&mut self) -> &mut Vec<f32> {
        match self {
            VectorStorage::Owned(v) => v,
            VectorStorage::Mapped(_) => unreachable!("call ensure_owned first"),
        }
    }
}

unsafe impl Send for VectorStorage {}
