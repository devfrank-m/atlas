use serde::{Deserialize, Serialize};

use crate::definitions::collections::Metric;
use crate::definitions::metadata::Metadata;

pub mod vector_storage;
pub mod vectors;
pub mod wal;

mod store;
mod worker;

pub use store::CollectionStore;
pub use vector_storage::VectorStorage;
pub use wal::WalRecord;
pub use worker::persistence_worker;

// ---------------------------------------------------------------------------
// In-memory snapshot types (returned by Index::snapshot, used by Collection)
// ---------------------------------------------------------------------------

pub struct CollectionData {
    pub id: String,
    pub name: String,
    pub dimension: usize,
    pub metric: Metric,
    pub index: IndexSnapshot,
    pub metadata: Vec<Metadata>,
    pub external_ids: Option<Vec<String>>,
}

pub enum IndexSnapshot {
    Flat {
        vectors: Vec<Vec<f32>>,
    },
    Hnsw {
        vectors: VectorStorage,
        nodes: Vec<NodeSnapshot>,
        entry_point: Option<usize>,
        max_layer: usize,
        m: usize,
        m_max0: usize,
        ef_construction: usize,
        ef_search: usize,
        level_mult: f64,
    },
}

// ---------------------------------------------------------------------------
// Disk snapshot types (serialized to JSON)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
pub struct VectorBlobRef {
    pub file: String,
    pub count: u64,
    pub dimension: u32,
    pub offset_bytes: u64,
}

#[derive(Serialize, Deserialize)]
pub struct SnapshotHeader {
    pub version: u32,
    pub generation: u64,
    pub index_type: String,
    pub vector_count: usize,
    pub dimension: usize,
    pub checksum: String,
}

#[derive(Serialize, Deserialize)]
pub struct CollectionSnapshot {
    pub header: SnapshotHeader,
    pub data: Box<serde_json::value::RawValue>,
}

#[derive(Serialize, Deserialize)]
pub struct DiskCollectionData {
    pub id: String,
    pub name: String,
    pub dimension: usize,
    pub metric: Metric,
    pub index: DiskIndexSnapshot,
    pub metadata: Vec<Metadata>,
    pub external_ids: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DiskIndexSnapshot {
    Flat {
        vectors: Vec<Vec<f32>>,
    },
    Hnsw {
        vectors: VectorBlobRef,
        nodes: Vec<NodeSnapshot>,
        entry_point: Option<usize>,
        max_layer: usize,
        m: usize,
        m_max0: usize,
        ef_construction: usize,
        ef_search: usize,
        level_mult: f64,
    },
}

impl DiskIndexSnapshot {
    pub fn index_type(&self) -> &'static str {
        match self {
            DiskIndexSnapshot::Flat { .. } => "flat",
            DiskIndexSnapshot::Hnsw { .. } => "hnsw",
        }
    }

    pub fn vector_count(&self) -> usize {
        match self {
            DiskIndexSnapshot::Flat { vectors } => vectors.len(),
            DiskIndexSnapshot::Hnsw { vectors, .. } => vectors.count as usize,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared sub-types (used by both IndexSnapshot and DiskIndexSnapshot)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
pub struct NodeSnapshot {
    pub connections: Vec<Vec<NeighborSnapshot>>,
}

#[derive(Serialize, Deserialize)]
pub struct NeighborSnapshot {
    pub id: usize,
    pub distance: f32,
}
