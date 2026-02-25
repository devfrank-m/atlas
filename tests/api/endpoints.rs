use atlas::api::{AppState, AppStateInner, build_router};
use axum::http::StatusCode;
use axum_test::TestServer;
use serde_json::json;
use std::sync::Arc;

fn test_server() -> TestServer {
    let state: AppState = Arc::new(AppStateInner::new());
    TestServer::new(build_router(state)).unwrap()
}

#[tokio::test]
async fn test_create_collection() {
    let server = test_server();
    let resp = server
        .post("/collections")
        .json(&json!({"name": "test", "dimension": 3, "metric": "cosine"}))
        .await;
    resp.assert_status(StatusCode::CREATED);
    let body: serde_json::Value = resp.json();
    assert_eq!(body["name"], "test");
    assert!(body["id"].as_str().unwrap().len() == 26);
}

#[tokio::test]
async fn test_create_collection_invalid_metric() {
    let server = test_server();
    let resp = server
        .post("/collections")
        .json(&json!({"name": "test", "dimension": 3, "metric": "invalid"}))
        .await;
    resp.assert_status(StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_get_collection() {
    let server = test_server();
    let create_resp = server
        .post("/collections")
        .json(&json!({"name": "planets", "dimension": 2, "metric": "cosine"}))
        .await;
    let id = create_resp.json::<serde_json::Value>()["id"]
        .as_str()
        .unwrap()
        .to_string();

    let resp = server.get(&format!("/collections/{id}")).await;
    resp.assert_status(StatusCode::OK);
    let body: serde_json::Value = resp.json();
    assert_eq!(body["name"], "planets");
    assert_eq!(body["dimension"], 2);
    assert_eq!(body["metric"], "cosine");
    assert_eq!(body["vector_count"], 0);
}

#[tokio::test]
async fn test_get_collection_not_found() {
    let server = test_server();
    let resp = server.get("/collections/00000000000000000000000000").await;
    resp.assert_status(StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_get_collection_invalid_id() {
    let server = test_server();
    let resp = server.get("/collections/not-a-ulid").await;
    resp.assert_status(StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_insert_and_search() {
    let server = test_server();
    let create_resp = server
        .post("/collections")
        .json(&json!({"name": "test", "dimension": 3, "metric": "cosine"}))
        .await;
    let id = create_resp.json::<serde_json::Value>()["id"]
        .as_str()
        .unwrap()
        .to_string();

    let resp = server
        .post(&format!("/collections/{id}/vectors"))
        .json(&json!({"vector": [1.0, 0.0, 0.0], "metadata": {"label": "x-axis"}}))
        .await;
    resp.assert_status(StatusCode::CREATED);
    assert_eq!(resp.json::<serde_json::Value>()["id"], 0);

    let resp = server
        .post(&format!("/collections/{id}/vectors"))
        .json(&json!({"vector": [0.0, 1.0, 0.0], "metadata": {"label": "y-axis"}}))
        .await;
    resp.assert_status(StatusCode::CREATED);
    assert_eq!(resp.json::<serde_json::Value>()["id"], 1);

    let resp = server
        .post(&format!("/collections/{id}/search"))
        .json(&json!({"vector": [1.0, 0.0, 0.0], "k": 2}))
        .await;
    resp.assert_status(StatusCode::OK);
    let body: serde_json::Value = resp.json();
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["metadata"]["label"], "x-axis");
}

#[tokio::test]
async fn test_insert_wrong_dimension() {
    let server = test_server();
    let create_resp = server
        .post("/collections")
        .json(&json!({"name": "test", "dimension": 3, "metric": "cosine"}))
        .await;
    let id = create_resp.json::<serde_json::Value>()["id"]
        .as_str()
        .unwrap()
        .to_string();

    let resp = server
        .post(&format!("/collections/{id}/vectors"))
        .json(&json!({"vector": [1.0, 0.0], "metadata": {}}))
        .await;
    resp.assert_status(StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_insert_collection_not_found() {
    let server = test_server();
    let resp = server
        .post("/collections/00000000000000000000000000/vectors")
        .json(&json!({"vector": [1.0], "metadata": {}}))
        .await;
    resp.assert_status(StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_search_wrong_dimension() {
    let server = test_server();
    let create_resp = server
        .post("/collections")
        .json(&json!({"name": "test", "dimension": 3, "metric": "cosine"}))
        .await;
    let id = create_resp.json::<serde_json::Value>()["id"]
        .as_str()
        .unwrap()
        .to_string();

    let resp = server
        .post(&format!("/collections/{id}/search"))
        .json(&json!({"vector": [1.0], "k": 1}))
        .await;
    resp.assert_status(StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_search_collection_not_found() {
    let server = test_server();
    let resp = server
        .post("/collections/00000000000000000000000000/search")
        .json(&json!({"vector": [1.0], "k": 1}))
        .await;
    resp.assert_status(StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_get_collection_after_insert() {
    let server = test_server();
    let create_resp = server
        .post("/collections")
        .json(&json!({"name": "test", "dimension": 2, "metric": "euclidean"}))
        .await;
    let id = create_resp.json::<serde_json::Value>()["id"]
        .as_str()
        .unwrap()
        .to_string();

    server
        .post(&format!("/collections/{id}/vectors"))
        .json(&json!({"vector": [1.0, 2.0], "metadata": {}}))
        .await;
    server
        .post(&format!("/collections/{id}/vectors"))
        .json(&json!({"vector": [3.0, 4.0], "metadata": {}}))
        .await;

    let resp = server.get(&format!("/collections/{id}")).await;
    let body: serde_json::Value = resp.json();
    assert_eq!(body["vector_count"], 2);
    assert_eq!(body["metric"], "euclidean");
}

#[tokio::test]
async fn test_insert_with_external_id() {
    let server = test_server();
    let create_resp = server
        .post("/collections")
        .json(&json!({"name": "test", "dimension": 2, "metric": "dot_product"}))
        .await;
    let id = create_resp.json::<serde_json::Value>()["id"]
        .as_str()
        .unwrap()
        .to_string();

    let resp = server
        .post(&format!("/collections/{id}/vectors"))
        .json(&json!({"vector": [1.0, 2.0], "metadata": {}, "external_id": "ext-1"}))
        .await;
    resp.assert_status(StatusCode::CREATED);

    let resp = server
        .post(&format!("/collections/{id}/search"))
        .json(&json!({"vector": [1.0, 2.0], "k": 1}))
        .await;
    let body: serde_json::Value = resp.json();
    assert_eq!(body["results"][0]["external_id"], "ext-1");
}
