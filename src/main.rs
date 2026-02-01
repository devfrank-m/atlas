use atlas::api::{AppState, AppStateInner, build_router};
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
    let app = build_router(state).layer(TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8600").await.unwrap();
    info!("Server running on http://0.0.0.0:8600");
    axum::serve(listener, app).await.unwrap();
}
