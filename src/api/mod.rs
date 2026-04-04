pub mod collections;
pub mod schemas;

use crate::api::schemas::HealthResponse;
use crate::definitions::collections::Collection;
use crate::persistence::CollectionStore;
use axum::{
    Json,
    Router,
    routing::{get, post},
};
use axum::http::StatusCode;
use collections::{create_collection, get_collection, insert_vector, search_collection};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use ulid::Ulid;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

pub type AppState = Arc<AppStateInner>;

pub enum PersistEvent {
    Dirty(Ulid),
    Delete(Ulid),
}

pub struct AppStateInner {
    pub collections: Mutex<HashMap<Ulid, Collection>>,
    pub store: CollectionStore,
    pub persist_tx: mpsc::Sender<PersistEvent>,
}

impl AppStateInner {
    pub fn new_with_persistence(
        data_dir: &str,
    ) -> std::io::Result<(Self, mpsc::Receiver<PersistEvent>)> {
        let store = CollectionStore::new(data_dir)?;
        let collections = store
            .load_all()
            .map_err(std::io::Error::other)?
            .into_iter()
            .map(|c| (c.id, c))
            .collect();
        let (persist_tx, persist_rx) = mpsc::channel(256);
        Ok((
            Self {
                collections: Mutex::new(collections),
                store,
                persist_tx,
            },
            persist_rx,
        ))
    }

    pub fn new_for_testing() -> Self {
        let tmp = std::env::temp_dir().join(format!("atlas_test_{}", Ulid::new()));
        let store = CollectionStore::new(tmp).expect("failed to create test store");
        let (persist_tx, _) = mpsc::channel(256);
        Self {
            collections: Mutex::new(HashMap::new()),
            store,
            persist_tx,
        }
    }
}

#[derive(OpenApi)]
#[openapi(
    paths(
        health,
        collections::create_collection,
        collections::get_collection,
        collections::insert_vector,
        collections::search_collection,
    ),
    components(schemas(
        schemas::HealthResponse,
        schemas::CollectionCreateRequest,
        schemas::CollectionCreateResponse,
        schemas::CollectionDetailResponse,
        schemas::VectorInsertRequest,
        schemas::VectorInsertResponse,
        schemas::SearchRequest,
        schemas::SearchResultResponse,
        schemas::SearchResponse,
        crate::definitions::filter::Filter,
    ))
)]
pub struct ApiDoc;

#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Service health", body = HealthResponse),
    )
)]
pub async fn health() -> (StatusCode, Json<HealthResponse>) {
    (
        StatusCode::OK,
        Json(HealthResponse {
            status: "ok".to_string(),
        }),
    )
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/collections", post(create_collection))
        .route("/collections/{id}", get(get_collection))
        .route("/collections/{id}/vectors", post(insert_vector))
        .route("/collections/{id}/search", post(search_collection))
        .with_state(state)
        .merge(SwaggerUi::new("/swagger-ui").url("/api-doc/openapi.json", ApiDoc::openapi()))
}
