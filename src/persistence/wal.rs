use serde::{Deserialize, Serialize};

use crate::definitions::metadata::Metadata;

#[derive(Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum WalRecord {
    Insert {
        vector: Vec<f32>,
        metadata: Metadata,
        external_id: Option<String>,
    },
}
