use crate::helpers::meta;
use atlas::definitions::collections::{Collection, IndexType, Metric};
use serde_json::json;
use std::collections::HashMap;
use ulid::Ulid;

#[test]
fn test_new_collection() {
    let col = Collection::new("test".to_string(), 3, Metric::Cosine, IndexType::Flat);
    let results = col.search(vec![1.0, 0.0, 0.0], 10, None);
    // empty collection returns no results
    assert_eq!(results.len(), 0);
}

#[test]
fn test_collection_has_unique_id() {
    let col1 = Collection::new("test".to_string(), 3, Metric::Cosine, IndexType::Flat);
    let col2 = Collection::new("test".to_string(), 3, Metric::Cosine, IndexType::Flat);
    // each collection gets a unique ULID
    assert_ne!(col1.id, col2.id);
    // id should be a valid ULID string
    assert!(col1.id.to_string().parse::<Ulid>().is_ok());
}

#[test]
fn test_insert_and_search_cosine() {
    let mut col = Collection::new("test".to_string(), 3, Metric::Cosine, IndexType::Flat);
    col.insert(vec![1.0, 0.0, 0.0], meta("x-axis"), None);
    col.insert(vec![0.0, 1.0, 0.0], meta("y-axis"), None);
    col.insert(vec![1.0, 1.0, 0.0], meta("diagonal"), None);

    let results = col.search(vec![1.0, 0.0, 0.0], 3, None);

    // all 3 vectors returned
    assert_eq!(results.len(), 3);
    // most similar to x-axis query should be the x-axis vector
    assert_eq!(results[0].metadata["label"], json!("x-axis"));
    // x-axis vs x-axis => cosine similarity of 1.0
    assert!(
        (results[0].score - 1.0).abs() < 1e-6,
        "identical vectors should have score 1.0"
    );
    // x-axis vs y-axis => orthogonal, score 0.0
    assert_eq!(results[2].metadata["label"], json!("y-axis"));
    assert!(
        (results[2].score - 0.0).abs() < 1e-6,
        "orthogonal vectors should have score 0.0"
    );
}

#[test]
fn test_insert_and_search_dot_product() {
    let mut col = Collection::new("test".to_string(), 2, Metric::DotProduct, IndexType::Flat);
    col.insert(vec![1.0, 2.0], meta("a"), None);
    col.insert(vec![3.0, 4.0], meta("b"), None);

    let results = col.search(vec![1.0, 1.0], 2, None);

    // dot(1,1)·(3,4) = 7, dot(1,1)·(1,2) = 3 — "b" ranked first
    assert_eq!(results[0].metadata["label"], json!("b"));
    assert!(
        (results[0].score - 7.0).abs() < 1e-6,
        "dot product of [1,1]·[3,4] should be 7.0"
    );
    assert_eq!(results[1].metadata["label"], json!("a"));
    assert!(
        (results[1].score - 3.0).abs() < 1e-6,
        "dot product of [1,1]·[1,2] should be 3.0"
    );
}

#[test]
fn test_insert_and_search_euclidean() {
    let mut col = Collection::new("test".to_string(), 2, Metric::Euclidean, IndexType::Flat);
    col.insert(vec![1.0, 0.0], meta("near"), None);
    col.insert(vec![10.0, 10.0], meta("far"), None);

    let results = col.search(vec![1.0, 0.0], 2, None);

    // nearest vector should rank first (higher score = closer)
    assert_eq!(results[0].metadata["label"], json!("near"));
    // distance 0 => score 1/(1+0) = 1.0
    assert!(
        (results[0].score - 1.0).abs() < 1e-6,
        "zero distance should produce score 1.0"
    );
    // far vector should have a lower score
    assert!(
        results[1].score < results[0].score,
        "farther vector should have a lower score"
    );
}

#[test]
fn test_search_k_limits_results() {
    let mut col = Collection::new("test".to_string(), 2, Metric::Cosine, IndexType::Flat);
    for i in 0..10 {
        col.insert(vec![i as f32, 1.0], meta(&format!("vec_{}", i)), None);
    }

    let results = col.search(vec![1.0, 0.0], 3, None);
    // only top-3 returned even though 10 vectors exist
    assert_eq!(results.len(), 3);
}

#[test]
fn test_search_results_sorted_descending() {
    let mut col = Collection::new("test".to_string(), 2, Metric::Cosine, IndexType::Flat);
    col.insert(vec![0.0, 1.0], meta("orthogonal"), None);
    col.insert(vec![1.0, 0.0], meta("identical"), None);
    col.insert(vec![1.0, 1.0], meta("diagonal"), None);

    let results = col.search(vec![1.0, 0.0], 3, None);

    // scores should be in descending order
    for i in 0..results.len() - 1 {
        assert!(
            results[i].score >= results[i + 1].score,
            "results should be sorted by score descending"
        );
    }
}

#[test]
fn test_search_result_contains_correct_vector() {
    let mut col = Collection::new("test".to_string(), 3, Metric::Cosine, IndexType::Flat);
    let v = vec![1.0, 2.0, 3.0];
    col.insert(v.clone(), meta("target"), None);

    let results = col.search(vec![1.0, 2.0, 3.0], 1, None);
    // returned vector should match the inserted one
    assert_eq!(results[0].vector, Some(v));
    assert_eq!(results[0].id, 0, "first inserted vector should have id 0");
}

#[test]
#[should_panic(expected = "Vector dimension does not match")]
fn test_insert_wrong_dimension_panics() {
    let mut col = Collection::new("test".to_string(), 3, Metric::Cosine, IndexType::Flat);
    col.insert(vec![1.0, 2.0], HashMap::new(), None);
}

#[test]
#[should_panic(expected = "Query dimension does not match")]
fn test_search_wrong_dimension_panics() {
    let col = Collection::new("test".to_string(), 3, Metric::Cosine, IndexType::Flat);
    col.search(vec![1.0, 2.0], 1, None);
}

// Unit-level tests for collection operations

#[test]
fn test_create_collection_returns_id() {
    let col = Collection::new(
        "my_collection".to_string(),
        128,
        Metric::Cosine,
        IndexType::Flat,
    );
    assert!(!col.id.to_string().is_empty());
    assert_eq!(col.name, "my_collection");
    assert_eq!(col.dimension, 128);
    assert_eq!(col.index.len(), 0);
}

#[test]
fn test_insert_vector_returns_sequential_ids() {
    let mut col = Collection::new("test".to_string(), 3, Metric::Cosine, IndexType::Flat);

    let id0 = col.index.len();
    col.insert(vec![1.0, 0.0, 0.0], meta("first"), None);
    assert_eq!(id0, 0);

    let id1 = col.index.len();
    col.insert(vec![0.0, 1.0, 0.0], meta("second"), None);
    assert_eq!(id1, 1);
}

#[test]
fn test_collection_detail_fields() {
    let mut col = Collection::new(
        "details_test".to_string(),
        3,
        Metric::Euclidean,
        IndexType::Flat,
    );
    col.insert(vec![1.0, 2.0, 3.0], meta("a"), None);
    col.insert(vec![4.0, 5.0, 6.0], meta("b"), None);

    assert_eq!(col.id.to_string().len(), 26);
    assert_eq!(col.name, "details_test");
    assert_eq!(col.dimension, 3);
    assert_eq!(col.index.len(), 2);
}

#[test]
fn test_search_response_structure() {
    let mut col = Collection::new(
        "search_test".to_string(),
        2,
        Metric::Cosine,
        IndexType::Flat,
    );
    col.insert(vec![1.0, 0.0], meta("alpha"), Some("ext-1".to_string()));
    col.insert(vec![0.0, 1.0], meta("beta"), Some("ext-2".to_string()));

    let results = col.search(vec![1.0, 0.0], 2, None);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].metadata["label"], json!("alpha"));
    assert_eq!(results[0].external_id, Some("ext-1".to_string()));
}

#[test]
fn test_search_response_without_external_ids() {
    let mut col = Collection::new(
        "search_test".to_string(),
        2,
        Metric::Cosine,
        IndexType::Flat,
    );
    col.insert(vec![1.0, 0.0], meta("alpha"), None);
    col.insert(vec![0.0, 1.0], meta("beta"), None);

    let results = col.search(vec![1.0, 0.0], 2, None);
    assert_eq!(results[0].external_id, None);
}

#[test]
fn test_collection_lookup_by_id() {
    let col1 = Collection::new("first".to_string(), 3, Metric::Cosine, IndexType::Flat);
    let col2 = Collection::new("second".to_string(), 3, Metric::Cosine, IndexType::Flat);
    let target_id = col2.id;

    let collections = vec![col1, col2];
    let found = collections.iter().find(|c| c.id == target_id);
    assert!(found.is_some());
    assert_eq!(found.unwrap().name, "second");
}
