use crate::definitions::filter::Filter;
use crate::definitions::metadata::Metadata;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
// --- Collection ---

#[derive(Serialize, Deserialize)]
pub struct CollectionCreateRequest {
    pub name: String,
    pub dimension: usize,
    pub metric: String,
    pub index: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct CollectionCreateResponse {
    pub id: String,
    pub name: String,
}

#[derive(Serialize, Deserialize)]
pub struct CollectionDetailResponse {
    pub id: String,
    pub name: String,
    pub dimension: usize,
    pub metric: String,
    pub vector_count: usize,
}

// --- Vector ---

#[derive(Serialize, Deserialize)]
pub struct VectorInsertRequest {
    pub vector: Vec<f32>,
    pub metadata: HashMap<String, Value>,
    pub external_id: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct VectorInsertResponse {
    pub id: usize,
}

// --- Search ---

#[derive(Serialize, Deserialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub k: usize,
    pub filter: Option<Filter>,
}

#[derive(Serialize, Deserialize)]
pub struct SearchResultResponse {
    pub id: usize,
    pub score: f32,
    pub metadata: Metadata,
    pub vector: Option<Vec<f32>>,
    pub external_id: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultResponse>,
}
