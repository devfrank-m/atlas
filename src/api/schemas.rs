use serde::{Deserialize, Serialize};

// --- Collection ---

#[derive(Serialize, Deserialize)]
pub struct CollectionCreateRequest {
    pub name: String,
    pub dimension: usize,
    pub metric: String,
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
    pub metadata: String,
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
}

#[derive(Serialize, Deserialize)]
pub struct SearchResultResponse {
    pub id: usize,
    pub score: f32,
    pub text: String,
    pub vector: Option<Vec<f32>>,
    pub external_id: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultResponse>,
}
