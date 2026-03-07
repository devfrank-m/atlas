use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use sha2::{Digest, Sha256};

use crate::definitions::collections::Collection;
use crate::persistence::wal::WalRecord;

use super::{
    CollectionData, CollectionSnapshot, DiskCollectionData, DiskIndexSnapshot, IndexSnapshot,
    SnapshotHeader, vectors,
};

pub struct CollectionStore {
    data_dir: PathBuf,
}

impl CollectionStore {
    pub fn new(data_dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let data_dir = data_dir.into();
        fs::create_dir_all(&data_dir)?;
        Ok(Self { data_dir })
    }

    fn collection_path(&self, id: &str) -> PathBuf {
        self.data_dir.join(format!("{id}.json"))
    }

    fn wal_path(&self, id: &str) -> PathBuf {
        self.data_dir.join(format!("{id}.wal"))
    }

    fn read_generation(&self, id: &str) -> u64 {
        let path = self.collection_path(id);
        if !path.exists() {
            return 0;
        }
        fs::read_to_string(&path)
            .ok()
            .and_then(|json| serde_json::from_str::<CollectionSnapshot>(&json).ok())
            .map(|s| s.header.generation)
            .unwrap_or(0)
    }

    pub fn save(&self, collection: &Collection) -> Result<(), String> {
        let id = collection.id.to_string();
        let old_generation = self.read_generation(&id);
        let new_generation = old_generation + 1;
        let data = collection.to_snapshot();
        let disk_data = self.to_disk_data(data, &id, new_generation)?;
        self.write_snapshot(disk_data, &id, new_generation)?;
        let old_blob = self.data_dir.join(format!("{id}.{old_generation}.vec"));
        if old_blob.exists() {
            let _ = fs::remove_file(&old_blob);
        }
        Ok(())
    }

    fn to_disk_data(
        &self,
        data: CollectionData,
        id: &str,
        generation: u64,
    ) -> Result<DiskCollectionData, String> {
        let disk_index = match data.index {
            IndexSnapshot::Flat { vectors } => DiskIndexSnapshot::Flat { vectors },
            IndexSnapshot::Hnsw {
                vectors,
                nodes,
                entry_point,
                max_layer,
                m,
                m_max0,
                ef_construction,
                ef_search,
                level_mult,
            } => {
                let blob = vectors::write(
                    &self.data_dir,
                    id,
                    generation,
                    vectors.as_slice(),
                    data.dimension as u32,
                )?;
                DiskIndexSnapshot::Hnsw {
                    vectors: blob,
                    nodes,
                    entry_point,
                    max_layer,
                    m,
                    m_max0,
                    ef_construction,
                    ef_search,
                    level_mult,
                }
            }
        };

        Ok(DiskCollectionData {
            id: data.id,
            name: data.name,
            dimension: data.dimension,
            metric: data.metric,
            index: disk_index,
            metadata: data.metadata,
            external_ids: data.external_ids,
        })
    }

    fn from_disk_data(&self, data: DiskCollectionData) -> Result<CollectionData, String> {
        let index = match data.index {
            DiskIndexSnapshot::Flat { vectors } => IndexSnapshot::Flat { vectors },
            DiskIndexSnapshot::Hnsw {
                vectors: blob,
                nodes,
                entry_point,
                max_layer,
                m,
                m_max0,
                ef_construction,
                ef_search,
                level_mult,
            } => {
                let storage = vectors::load_mmap(&self.data_dir, &blob)?;
                IndexSnapshot::Hnsw {
                    vectors: storage,
                    nodes,
                    entry_point,
                    max_layer,
                    m,
                    m_max0,
                    ef_construction,
                    ef_search,
                    level_mult,
                }
            }
        };

        Ok(CollectionData {
            id: data.id,
            name: data.name,
            dimension: data.dimension,
            metric: data.metric,
            index,
            metadata: data.metadata,
            external_ids: data.external_ids,
        })
    }

    fn write_snapshot(
        &self,
        data: DiskCollectionData,
        id: &str,
        generation: u64,
    ) -> Result<(), String> {
        let index_type = data.index.index_type().to_string();
        let vector_count = data.index.vector_count();
        let dimension = data.dimension;

        let data_json = serde_json::to_string(&data).map_err(|e| e.to_string())?;
        let checksum = sha256_hex(data_json.as_bytes());
        let raw_data =
            serde_json::value::RawValue::from_string(data_json).map_err(|e| e.to_string())?;

        let snapshot = CollectionSnapshot {
            header: SnapshotHeader {
                version: 2,
                generation,
                index_type,
                vector_count,
                dimension,
                checksum,
            },
            data: raw_data,
        };

        let json = serde_json::to_string(&snapshot).map_err(|e| e.to_string())?;
        let path = self.collection_path(id);
        let tmp = path.with_extension("json.tmp");

        fs::write(&tmp, json.as_bytes()).map_err(|e| e.to_string())?;
        fs::rename(&tmp, &path).map_err(|e| e.to_string())
    }

    pub fn load(&self, id: &str) -> Result<Collection, String> {
        let path = self.collection_path(id);
        let json = fs::read_to_string(&path).map_err(|e| e.to_string())?;
        let snapshot: CollectionSnapshot =
            serde_json::from_str(&json).map_err(|e| e.to_string())?;

        if snapshot.header.version != 2 {
            return Err(format!(
                "unsupported snapshot version: {}",
                snapshot.header.version
            ));
        }

        verify_checksum(&snapshot)?;

        let disk_data: DiskCollectionData =
            serde_json::from_str(snapshot.data.get()).map_err(|e| e.to_string())?;
        let data = self.from_disk_data(disk_data)?;
        let mut collection = Collection::from_snapshot(data)?;

        if self.wal_path(id).exists() {
            self.wal_replay(id, &mut collection)?;
            self.wal_truncate(id)?;
            self.save(&collection)?;
        }

        Ok(collection)
    }

    pub fn load_all(&self) -> Result<Vec<Collection>, String> {
        let mut collections = Vec::new();
        let entries = fs::read_dir(&self.data_dir).map_err(|e| e.to_string())?;

        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }

            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or("invalid filename")?
                .to_string();

            match self.load(&stem) {
                Ok(col) => collections.push(col),
                Err(e) => tracing::warn!("skipping {stem}: {e}"),
            }
        }

        Ok(collections)
    }

    pub fn delete(&self, id: &str) -> Result<(), String> {
        for path in [self.collection_path(id), self.wal_path(id)] {
            if path.exists() {
                fs::remove_file(&path).map_err(|e| e.to_string())?;
            }
        }
        if let Ok(entries) = fs::read_dir(&self.data_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with(&format!("{id}.")) && name.ends_with(".vec") {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
        Ok(())
    }

    pub fn wal_append(&self, id: &str, record: &WalRecord) -> Result<(), String> {
        let mut line = serde_json::to_string(record).map_err(|e| e.to_string())?;
        line.push('\n');

        OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.wal_path(id))
            .and_then(|mut f| f.write_all(line.as_bytes()))
            .map_err(|e| e.to_string())
    }

    fn wal_replay(&self, id: &str, collection: &mut Collection) -> Result<(), String> {
        let content = fs::read_to_string(self.wal_path(id)).map_err(|e| e.to_string())?;

        for line in content.lines() {
            if line.is_empty() {
                continue;
            }
            let record: WalRecord = serde_json::from_str(line).map_err(|e| e.to_string())?;
            match record {
                WalRecord::Insert {
                    vector,
                    metadata,
                    external_id,
                } => collection.insert(vector, metadata, external_id),
            }
        }

        Ok(())
    }

    fn wal_truncate(&self, id: &str) -> Result<(), String> {
        let path = self.wal_path(id);
        if path.exists() {
            fs::remove_file(&path).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

fn verify_checksum(snapshot: &CollectionSnapshot) -> Result<(), String> {
    let expected = sha256_hex(snapshot.data.get().as_bytes());
    if expected != snapshot.header.checksum {
        return Err("checksum mismatch: file may be corrupted".into());
    }
    Ok(())
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}
