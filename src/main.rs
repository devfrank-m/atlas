use atlas::api::{AppState, AppStateInner, build_router};
use atlas::persistence::persistence_worker;
use atlas::settings;
use std::sync::Arc;
use tokio::sync::oneshot;
use tower_http::trace::TraceLayer;
use tracing::info;

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(
            "atlas=debug,tower_http=debug",
        ))
        .init();

    let data_dir = settings::SETTINGS.data_dir();
    let (state_inner, persist_rx) =
        AppStateInner::new_with_persistence(&data_dir).expect("failed to initialize data store");
    let state: AppState = Arc::new(state_inner);

    let (worker_shutdown_tx, worker_shutdown_rx) = oneshot::channel::<()>();
    let worker = tokio::spawn(persistence_worker(
        state.clone(),
        persist_rx,
        worker_shutdown_rx,
    ));

    let app = build_router(state).layer(TraceLayer::new_for_http());
    let listener = tokio::net::TcpListener::bind(format!(
        "{}:{}",
        settings::SETTINGS.host(),
        settings::SETTINGS.port()
    ))
    .await
    .expect("failed to bind to address");
    info!(
        "HTTP server listening on http://{}:{}",
        settings::SETTINGS.host(),
        settings::SETTINGS.port()
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();

    info!("HTTP server shut down, waiting for persistence worker");
    let _ = worker_shutdown_tx.send(());
    worker.await.unwrap();
    info!("Server exited cleanly");
}
