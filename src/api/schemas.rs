use crate::definitions::filter::Filter;
use crate::definitions::metadata::Metadata;
use serde::{Deserialize, Serialize};
// --- Collection ---

#[derive(Serialize, Deserialize, utoipa::ToSchema)]
pub struct CollectionCreateRequest {
    pub name: String,
    pub dimension: usize,
    pub metric: String,
    pub index: Option<String>,
}

#[derive(Serialize, Deserialize, utoipa::ToSchema)]
pub struct CollectionCreateResponse {
    pub id: String,
    pub name: String,
}

#[derive(Serialize, Deserialize, utoipa::ToSchema)]
pub struct CollectionDetailResponse {
    pub id: String,
    pub name: String,
    pub dimension: usize,
    pub metric: String,
    pub vector_count: usize,
}

// --- Vector ---

#[derive(Serialize, Deserialize, utoipa::ToSchema)]
pub struct VectorInsertRequest {
    pub vector: Vec<f32>,
    #[schema(value_type = Object)]
    pub metadata: Metadata,
    pub external_id: Option<String>,
}

#[derive(Serialize, Deserialize, utoipa::ToSchema)]
pub struct VectorInsertResponse {
    pub id: usize,
}

// --- Search ---

#[derive(Serialize, Deserialize, utoipa::ToSchema)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub k: usize,
    pub filter: Option<Filter>,
}

#[derive(Serialize, Deserialize, utoipa::ToSchema)]
pub struct SearchResultResponse {
    pub id: usize,
    pub score: f32,
    #[schema(value_type = Object)]
    pub metadata: Metadata,
    pub vector: Option<Vec<f32>>,
    pub external_id: Option<String>,
}

#[derive(Serialize, Deserialize, utoipa::ToSchema)]
pub struct SearchResponse {
    pub results: Vec<SearchResultResponse>,
}
