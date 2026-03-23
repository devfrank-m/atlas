use crate::api::schemas::{
    CollectionCreateRequest, CollectionCreateResponse, CollectionDetailResponse, SearchRequest,
    SearchResponse, SearchResultResponse, VectorInsertRequest, VectorInsertResponse,
};
use crate::api::{AppState, PersistEvent};
use crate::common::api::{bad_request, internal_server_error, not_found};
use crate::common::utils::parse_ulid;
use crate::definitions::collections::{Collection, IndexType, Metric};
use crate::definitions::errors::ValidationError;
use crate::persistence::WalRecord;
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

#[utoipa::path(
    post,
    path = "/collections",
    request_body = CollectionCreateRequest,
    responses(
        (status = 201, description = "Collection created", body = CollectionCreateResponse),
        (status = 400, description = "Invalid metric or index type"),
    )
)]
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
    let id = collection.id;

    state.collections.lock().unwrap().insert(id, collection);
    let _ = state.persist_tx.try_send(PersistEvent::Dirty(id));

    (StatusCode::CREATED, Json(resp)).into_response()
}

#[utoipa::path(
    get,
    path = "/collections/{id}",
    params(
        ("id" = String, Path, description = "Collection ULID"),
    ),
    responses(
        (status = 200, description = "Collection details", body = CollectionDetailResponse),
        (status = 400, description = "Invalid ULID"),
        (status = 404, description = "Collection not found"),
    )
)]
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
    match collections.get(&ulid) {
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
        None => not_found("Collection not found"),
    }
}

#[utoipa::path(
    post,
    path = "/collections/{id}/vectors",
    params(
        ("id" = String, Path, description = "Collection ULID"),
    ),
    request_body = VectorInsertRequest,
    responses(
        (status = 201, description = "Vector inserted", body = VectorInsertResponse),
        (status = 400, description = "Invalid ULID or dimension mismatch"),
        (status = 404, description = "Collection not found"),
        (status = 500, description = "Persistence error"),
    )
)]
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
    match collections.get_mut(&ulid) {
        Some(col) => {
            if req.vector.len() != col.dimension {
                return bad_request(format!(
                    "Vector dimension {} does not match collection dimension {}",
                    req.vector.len(),
                    col.dimension
                ));
            }
            let vector_id = col.index.len();

            if let Err(e) = state.store.wal_append(
                &ulid.to_string(),
                &WalRecord::Insert {
                    vector: req.vector.clone(),
                    metadata: req.metadata.clone(),
                    external_id: req.external_id.clone(),
                },
            ) {
                tracing::error!("WAL append failed for {ulid}: {e}");
                return internal_server_error("Failed to persist write");
            }

            col.insert(req.vector, req.metadata, req.external_id);
            let _ = state.persist_tx.try_send(PersistEvent::Dirty(ulid));
            (
                StatusCode::CREATED,
                Json(VectorInsertResponse { id: vector_id }),
            )
                .into_response()
        }
        None => not_found("Collection not found"),
    }
}

#[utoipa::path(
    post,
    path = "/collections/{id}/search",
    params(
        ("id" = String, Path, description = "Collection ULID"),
    ),
    request_body = SearchRequest,
    responses(
        (status = 200, description = "Search results", body = SearchResponse),
        (status = 400, description = "Invalid ULID or dimension mismatch"),
        (status = 404, description = "Collection not found"),
    )
)]
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
    match collections.get(&ulid) {
        Some(col) => {
            if req.vector.len() != col.dimension {
                return bad_request(format!(
                    "Query dimension {} does not match collection dimension {}",
                    req.vector.len(),
                    col.dimension
                ));
            }
            let results = col.search(req.vector, req.k, req.filter);
            let resp = SearchResponse {
                results: results
                    .into_iter()
                    .map(|r| SearchResultResponse {
                        id: r.id,
                        score: r.score,
                        vector: r.vector,
                        external_id: r.external_id,
                        metadata: r.metadata,
                    })
                    .collect(),
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        None => not_found("Collection not found"),
    }
}
