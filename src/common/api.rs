use axum::Json;
use axum::http::StatusCode;
use axum::response::IntoResponse;

pub fn bad_request(msg: impl Into<String>) -> axum::response::Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error": msg.into()})),
    )
        .into_response()
}

pub fn not_found(msg: impl Into<String>) -> axum::response::Response {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({"error": msg.into()})),
    )
        .into_response()
}

pub fn internal_server_error(msg: impl Into<String>) -> axum::response::Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({"error": msg.into()})),
    )
        .into_response()
}
