pub mod collections;
pub mod schemas;

use crate::definitions::collections::Collection;
use axum::{
    Router,
    routing::{get, post},
};
use collections::{create_collection, get_collection, insert_vector, search_collection};
use std::sync::{Arc, Mutex};

pub type AppState = Arc<AppStateInner>;

pub struct AppStateInner {
    pub collections: Mutex<Vec<Collection>>,
}

impl Default for AppStateInner {
    fn default() -> Self {
        Self::new()
    }
}

impl AppStateInner {
    pub fn new() -> Self {
        Self {
            collections: Mutex::new(Vec::new()),
        }
    }
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/collections", post(create_collection))
        .route("/collections/{id}", get(get_collection))
        .route("/collections/{id}/vectors", post(insert_vector))
        .route("/collections/{id}/search", post(search_collection))
        .with_state(state)
}
