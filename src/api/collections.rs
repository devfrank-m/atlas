use crate::api::AppState;
use crate::api::schemas::{
    CollectionCreateRequest, CollectionCreateResponse, CollectionDetailResponse, SearchRequest,
    SearchResponse, SearchResultResponse, VectorInsertRequest, VectorInsertResponse,
};
use crate::common::api::bad_request;
use crate::common::utils::parse_ulid;
use crate::definitions::collections::{Collection, IndexType, Metric};
use crate::definitions::errors::ValidationError;
use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};

fn parse_metric(s: &str) -> Option<Metric> {
    match s.to_lowercase().as_str() {
        "cosine" => Some(Metric::Cosine),
        "euclidean" => Some(Metric::Euclidean),
        "dot_product" | "dotproduct" => Some(Metric::DotProduct),
        _ => None,
    }
}

fn metric_to_string(m: &Metric) -> &'static str {
    match m {
        Metric::Cosine => "cosine",
        Metric::Euclidean => "euclidean",
        Metric::DotProduct => "dot_product",
    }
}

fn parse_index(s: &str) -> Result<IndexType, ValidationError> {
    match s.to_lowercase().as_str() {
        "flat" => Ok(IndexType::Flat),
        "hnsw" => Ok(IndexType::Hnsw),
        _ => Err(ValidationError::new(
            "Invalid index type. Use: flat, hnsw".to_string(),
        )),
    }
}

pub async fn create_collection(
    State(state): State<AppState>,
    Json(req): Json<CollectionCreateRequest>,
) -> impl IntoResponse {
    let metric = match parse_metric(&req.metric) {
        Some(m) => m,
        None => {
            return bad_request("Invalid metric. Use: cosine, euclidean, dot_product".to_string());
        }
    };

    let index_type = match parse_index(&req.index.unwrap_or("flat".to_string())) {
        Ok(i) => i,
        Err(e) => return bad_request(e.to_string()),
    };

    let collection = Collection::new(req.name.clone(), req.dimension, metric, index_type);
    let resp = CollectionCreateResponse {
        id: collection.id.to_string(),
        name: collection.name.clone(),
    };

    state.collections.lock().unwrap().push(collection);

    (StatusCode::CREATED, Json(resp)).into_response()
}

pub async fn get_collection(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let ulid = match parse_ulid(&id) {
        Ok(u) => u,
        Err(e) => {
            return bad_request(e.to_string());
        }
    };

    let collections = state.collections.lock().unwrap();
    match collections.iter().find(|c| c.id == ulid) {
        Some(col) => {
            let resp = CollectionDetailResponse {
                id: col.id.to_string(),
                name: col.name.clone(),
                dimension: col.dimension,
                metric: metric_to_string(&col.metric).to_string(),
                vector_count: col.index.len(),
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Collection not found"})),
        )
            .into_response(),
    }
}

pub async fn insert_vector(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<VectorInsertRequest>,
) -> impl IntoResponse {
    let ulid = match parse_ulid(&id) {
        Ok(u) => u,
        Err(e) => {
            return bad_request(e.to_string());
        }
    };

    let mut collections = state.collections.lock().unwrap();
    match collections.iter_mut().find(|c| c.id == ulid) {
        Some(col) => {
            if req.vector.len() != col.dimension {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": format!("Vector dimension {} does not match collection dimension {}", req.vector.len(), col.dimension)
                    })),
                )
                    .into_response();
            }
            let vector_id = col.index.len();
            col.insert(req.vector, req.metadata, req.external_id);
            (
                StatusCode::CREATED,
                Json(VectorInsertResponse { id: vector_id }),
            )
                .into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Collection not found"})),
        )
            .into_response(),
    }
}

pub async fn search_collection(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<SearchRequest>,
) -> impl IntoResponse {
    let ulid = match parse_ulid(&id) {
        Ok(u) => u,
        Err(e) => {
            return bad_request(e.to_string());
        }
    };
    let collections = state.collections.lock().unwrap();
    match collections.iter().find(|c| c.id == ulid) {
        Some(col) => {
            if req.vector.len() != col.dimension {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": format!("Query dimension {} does not match collection dimension {}", req.vector.len(), col.dimension)
                    })),
                )
                    .into_response();
            }
            let results = col.search(req.vector, req.k);
            let resp = SearchResponse {
                results: results
                    .into_iter()
                    .map(|r| SearchResultResponse {
                        id: r.id,
                        score: r.score,
                        text: r.text,
                        vector: r.vector,
                        external_id: r.external_id,
                        metadata: r.metadata,
                    })
                    .collect(),
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Collection not found"})),
        )
            .into_response(),
    }
}
