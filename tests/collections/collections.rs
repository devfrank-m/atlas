use atlas::definitions::collections::{Collection, Metric};

#[test]
fn test_new_collection() {
    let col = Collection::new("test".to_string(), 3, Metric::Cosine);
    let results = col.search(vec![1.0, 0.0, 0.0], 10);
    // empty collection returns no results
    assert_eq!(results.len(), 0);
}

#[test]
fn test_insert_and_search_cosine() {
    let mut col = Collection::new("test".to_string(), 3, Metric::Cosine);
    col.insert(vec![1.0, 0.0, 0.0], "x-axis".to_string(), None);
    col.insert(vec![0.0, 1.0, 0.0], "y-axis".to_string(), None);
    col.insert(vec![1.0, 1.0, 0.0], "diagonal".to_string(), None);

    let results = col.search(vec![1.0, 0.0, 0.0], 3);

    // all 3 vectors returned
    assert_eq!(results.len(), 3);
    // most similar to x-axis query should be the x-axis vector
    assert_eq!(results[0].text, "x-axis");
    // x-axis vs x-axis => cosine similarity of 1.0
    assert!(
        (results[0].score - 1.0).abs() < 1e-6,
        "identical vectors should have score 1.0"
    );
    // x-axis vs y-axis => orthogonal, score 0.0
    assert_eq!(results[2].text, "y-axis");
    assert!(
        (results[2].score - 0.0).abs() < 1e-6,
        "orthogonal vectors should have score 0.0"
    );
}

#[test]
fn test_insert_and_search_dot_product() {
    let mut col = Collection::new("test".to_string(), 2, Metric::DotProduct);
    col.insert(vec![1.0, 2.0], "a".to_string(), None);
    col.insert(vec![3.0, 4.0], "b".to_string(), None);

    let results = col.search(vec![1.0, 1.0], 2);

    // dot(1,1)·(3,4) = 7, dot(1,1)·(1,2) = 3 — "b" ranked first
    assert_eq!(results[0].text, "b");
    assert!(
        (results[0].score - 7.0).abs() < 1e-6,
        "dot product of [1,1]·[3,4] should be 7.0"
    );
    assert_eq!(results[1].text, "a");
    assert!(
        (results[1].score - 3.0).abs() < 1e-6,
        "dot product of [1,1]·[1,2] should be 3.0"
    );
}

#[test]
fn test_insert_and_search_euclidean() {
    let mut col = Collection::new("test".to_string(), 2, Metric::Euclidean);
    col.insert(vec![1.0, 0.0], "near".to_string(), None);
    col.insert(vec![10.0, 10.0], "far".to_string(), None);

    let results = col.search(vec![1.0, 0.0], 2);

    // nearest vector should rank first (higher score = closer)
    assert_eq!(results[0].text, "near");
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
    let mut col = Collection::new("test".to_string(), 2, Metric::Cosine);
    for i in 0..10 {
        col.insert(vec![i as f32, 1.0], format!("vec_{}", i), None);
    }

    let results = col.search(vec![1.0, 0.0], 3);
    // only top-3 returned even though 10 vectors exist
    assert_eq!(results.len(), 3);
}

#[test]
fn test_search_results_sorted_descending() {
    let mut col = Collection::new("test".to_string(), 2, Metric::Cosine);
    col.insert(vec![0.0, 1.0], "orthogonal".to_string(), None);
    col.insert(vec![1.0, 0.0], "identical".to_string(), None);
    col.insert(vec![1.0, 1.0], "diagonal".to_string(), None);

    let results = col.search(vec![1.0, 0.0], 3);

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
    let mut col = Collection::new("test".to_string(), 3, Metric::Cosine);
    let v = vec![1.0, 2.0, 3.0];
    col.insert(v.clone(), "target".to_string(), None);

    let results = col.search(vec![1.0, 2.0, 3.0], 1);
    // returned vector should match the inserted one
    assert_eq!(results[0].vector, v);
    assert_eq!(results[0].id, 0, "first inserted vector should have id 0");
}

#[test]
#[should_panic(expected = "Vector dimension does not match")]
fn test_insert_wrong_dimension_panics() {
    let mut col = Collection::new("test".to_string(), 3, Metric::Cosine);
    col.insert(vec![1.0, 2.0], "bad".to_string(), None);
}

#[test]
#[should_panic(expected = "Query dimension does not match")]
fn test_search_wrong_dimension_panics() {
    let col = Collection::new("test".to_string(), 3, Metric::Cosine);
    col.search(vec![1.0, 2.0], 1);
}
