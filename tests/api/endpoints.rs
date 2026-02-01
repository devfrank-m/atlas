use atlas::api::{AppState, AppStateInner, build_router};
use atlas::definitions::collections::{Collection, Metric};
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
        .json(&json!({"vector": [1.0, 0.0, 0.0], "metadata": "x-axis"}))
        .await;
    resp.assert_status(StatusCode::CREATED);
    assert_eq!(resp.json::<serde_json::Value>()["id"], 0);

    let resp = server
        .post(&format!("/collections/{id}/vectors"))
        .json(&json!({"vector": [0.0, 1.0, 0.0], "metadata": "y-axis"}))
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
    assert_eq!(results[0]["text"], "x-axis");
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
        .json(&json!({"vector": [1.0, 0.0], "metadata": "bad"}))
        .await;
    resp.assert_status(StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_insert_collection_not_found() {
    let server = test_server();
    let resp = server
        .post("/collections/00000000000000000000000000/vectors")
        .json(&json!({"vector": [1.0], "metadata": "x"}))
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
        .json(&json!({"vector": [1.0, 2.0], "metadata": "a"}))
        .await;
    server
        .post(&format!("/collections/{id}/vectors"))
        .json(&json!({"vector": [3.0, 4.0], "metadata": "b"}))
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
        .json(&json!({"vector": [1.0, 2.0], "metadata": "a", "external_id": "ext-1"}))
        .await;
    resp.assert_status(StatusCode::CREATED);

    let resp = server
        .post(&format!("/collections/{id}/search"))
        .json(&json!({"vector": [1.0, 2.0], "k": 1}))
        .await;
    let body: serde_json::Value = resp.json();
    assert_eq!(body["results"][0]["external_id"], "ext-1");
}

// Unit-level tests for collection operations

#[test]
fn test_create_collection_returns_id() {
    let col = Collection::new("my_collection".to_string(), 128, Metric::Cosine);
    assert!(!col.id.to_string().is_empty());
    assert_eq!(col.name, "my_collection");
    assert_eq!(col.dimension, 128);
    assert_eq!(col.vectors.len(), 0);
}

#[test]
fn test_insert_vector_returns_sequential_ids() {
    let mut col = Collection::new("test".to_string(), 3, Metric::Cosine);

    let id0 = col.vectors.len();
    col.insert(vec![1.0, 0.0, 0.0], "first".to_string(), None);
    assert_eq!(id0, 0);

    let id1 = col.vectors.len();
    col.insert(vec![0.0, 1.0, 0.0], "second".to_string(), None);
    assert_eq!(id1, 1);
}

#[test]
fn test_collection_detail_fields() {
    let mut col = Collection::new("details_test".to_string(), 3, Metric::Euclidean);
    col.insert(vec![1.0, 2.0, 3.0], "a".to_string(), None);
    col.insert(vec![4.0, 5.0, 6.0], "b".to_string(), None);

    assert_eq!(col.id.to_string().len(), 26);
    assert_eq!(col.name, "details_test");
    assert_eq!(col.dimension, 3);
    assert_eq!(col.vectors.len(), 2);
}

#[test]
fn test_search_response_structure() {
    let mut col = Collection::new("search_test".to_string(), 2, Metric::Cosine);
    col.insert(
        vec![1.0, 0.0],
        "alpha".to_string(),
        Some("ext-1".to_string()),
    );
    col.insert(
        vec![0.0, 1.0],
        "beta".to_string(),
        Some("ext-2".to_string()),
    );

    let results = col.search(vec![1.0, 0.0], 2);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].text, "alpha");
    assert_eq!(results[0].external_id, Some("ext-1".to_string()));
}

#[test]
fn test_search_response_without_external_ids() {
    let mut col = Collection::new("search_test".to_string(), 2, Metric::Cosine);
    col.insert(vec![1.0, 0.0], "alpha".to_string(), None);
    col.insert(vec![0.0, 1.0], "beta".to_string(), None);

    let results = col.search(vec![1.0, 0.0], 2);
    assert_eq!(results[0].external_id, None);
}

#[test]
fn test_collection_lookup_by_id() {
    let col1 = Collection::new("first".to_string(), 3, Metric::Cosine);
    let col2 = Collection::new("second".to_string(), 3, Metric::Cosine);
    let target_id = col2.id;

    let collections = vec![col1, col2];
    let found = collections.iter().find(|c| c.id == target_id);
    assert!(found.is_some());
    assert_eq!(found.unwrap().name, "second");
}
