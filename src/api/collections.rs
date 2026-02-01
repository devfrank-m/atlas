use axum::{Json, http::StatusCode, response::IntoResponse};
use crate::api::schemas::CollectionCreateRequest;

pub async fn create_collection(Json(create_request): Json<CollectionCreateRequest>) -> impl IntoResponse {
    (
        StatusCode::CREATED,
        Json(serde_json::json!({
            "message": "Collection created",
            "collection": create_request
        })),
    )
        .into_response()
}
