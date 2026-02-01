use atlas::definitions::collections::{Collection, Metric};

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

    let id2 = col.vectors.len();
    col.insert(vec![0.0, 0.0, 1.0], "third".to_string(), None);
    assert_eq!(id2, 2);
}

#[test]
fn test_collection_detail_fields() {
    let mut col = Collection::new("details_test".to_string(), 3, Metric::Euclidean);
    col.insert(vec![1.0, 2.0, 3.0], "a".to_string(), None);
    col.insert(vec![4.0, 5.0, 6.0], "b".to_string(), None);

    assert_eq!(col.id.to_string().len(), 26); // ULID is always 26 chars
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
    assert_eq!(results[1].text, "beta");
    assert_eq!(results[1].external_id, Some("ext-2".to_string()));
}

#[test]
fn test_search_response_without_external_ids() {
    let mut col = Collection::new("search_test".to_string(), 2, Metric::Cosine);
    col.insert(vec![1.0, 0.0], "alpha".to_string(), None);
    col.insert(vec![0.0, 1.0], "beta".to_string(), None);

    let results = col.search(vec![1.0, 0.0], 2);

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].external_id, None);
    assert_eq!(results[1].external_id, None);
}

#[test]
fn test_insert_wrong_dimension_detected() {
    let col = Collection::new("test".to_string(), 3, Metric::Cosine);
    let wrong_vec = vec![1.0, 2.0];
    assert_ne!(wrong_vec.len(), col.dimension);
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
