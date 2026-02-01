mod api;

use api::collections::{create_collection, get_collection, insert_vector, search_collection};
use api::{AppState, AppStateInner};
use axum::{
    Router,
    routing::{get, post},
};
use std::sync::Arc;
use tower_http::trace::TraceLayer;
use tracing::info;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(
            "atlas=debug,tower_http=debug",
        ))
        .init();

    let state: AppState = Arc::new(AppStateInner::new());

    let app = Router::new()
        .route("/collections", post(create_collection))
        .route("/collections/{id}", get(get_collection))
        .route("/collections/{id}/vectors", post(insert_vector))
        .route("/collections/{id}/search", post(search_collection))
        .with_state(state)
        .layer(TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8600").await.unwrap();
    info!("Server running on http://0.0.0.0:8600");
    axum::serve(listener, app).await.unwrap();
}
