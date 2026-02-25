use atlas::definitions::collections::Metric;
use atlas::index::base::Index;
use atlas::index::flat::FlatIndex;

fn new_flat(dim: usize, metric: Metric) -> FlatIndex {
    FlatIndex {
        vectors: Vec::new(),
        dimension: dim,
        metric,
    }
}

#[test]
fn test_empty_index() {
    let index = new_flat(3, Metric::Cosine);
    let results = index.search(vec![1.0, 0.0, 0.0], 10);
    assert_eq!(results.len(), 0);
}

#[test]
fn test_k_zero_returns_empty() {
    let mut index = new_flat(3, Metric::Cosine);
    index.insert(vec![1.0, 0.0, 0.0]);
    let results = index.search(vec![1.0, 0.0, 0.0], 0);
    assert_eq!(results.len(), 0);
}

#[test]
fn test_single_vector() {
    let mut index = new_flat(3, Metric::Cosine);
    index.insert(vec![1.0, 0.0, 0.0]);

    let results = index.search(vec![1.0, 0.0, 0.0], 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 0);
    assert!((results[0].score - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_search() {
    let mut index = new_flat(3, Metric::Cosine);
    index.insert(vec![1.0, 0.0, 0.0]);
    index.insert(vec![0.0, 1.0, 0.0]);
    index.insert(vec![1.0, 1.0, 0.0]);

    let results = index.search(vec![1.0, 0.0, 0.0], 3);

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].id, 0);
    assert!((results[0].score - 1.0).abs() < 1e-6);
    assert_eq!(results[2].id, 1);
    assert!((results[2].score - 0.0).abs() < 1e-6);
}

#[test]
fn test_dot_product_search() {
    let mut index = new_flat(2, Metric::DotProduct);
    index.insert(vec![1.0, 2.0]);
    index.insert(vec![3.0, 4.0]);

    let results = index.search(vec![1.0, 1.0], 2);

    assert_eq!(results[0].id, 1);
    assert!((results[0].score - 7.0).abs() < 1e-6);
    assert_eq!(results[1].id, 0);
    assert!((results[1].score - 3.0).abs() < 1e-6);
}

#[test]
fn test_euclidean_search() {
    let mut index = new_flat(2, Metric::Euclidean);
    index.insert(vec![1.0, 0.0]);
    index.insert(vec![10.0, 10.0]);

    let results = index.search(vec![1.0, 0.0], 2);

    assert_eq!(results[0].id, 0);
    assert!((results[0].score - 1.0).abs() < 1e-6);
    assert!(results[1].score < results[0].score);
}

#[test]
fn test_results_sorted_descending() {
    let mut index = new_flat(2, Metric::Cosine);
    index.insert(vec![0.0, 1.0]);
    index.insert(vec![1.0, 0.0]);
    index.insert(vec![1.0, 1.0]);

    let results = index.search(vec![1.0, 0.0], 3);

    for w in results.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "results should be sorted by score descending"
        );
    }
}

#[test]
fn test_k_larger_than_index() {
    let mut index = new_flat(2, Metric::Cosine);
    index.insert(vec![1.0, 0.0]);
    index.insert(vec![0.0, 1.0]);

    let results = index.search(vec![1.0, 0.0], 10);
    assert_eq!(results.len(), 2);
}

#[test]
fn test_get_returns_correct_vector() {
    let mut index = new_flat(3, Metric::Cosine);
    let v = vec![1.0, 2.0, 3.0];
    index.insert(v.clone());

    assert_eq!(index.get(0), Some(v.as_slice()));
    assert_eq!(index.get(1), None);
}

#[test]
fn test_len_and_is_empty() {
    let mut index = new_flat(2, Metric::Cosine);
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);

    index.insert(vec![1.0, 0.0]);
    assert!(!index.is_empty());
    assert_eq!(index.len(), 1);

    index.insert(vec![0.0, 1.0]);
    assert_eq!(index.len(), 2);
}

#[test]
#[should_panic(expected = "Query dimension does not match")]
fn test_search_wrong_dimension_panics() {
    let index = new_flat(3, Metric::Cosine);
    index.search(vec![1.0, 2.0], 1);
}

#[test]
fn test_many_inserts() {
    let mut index = new_flat(4, Metric::Cosine);
    for i in 0..500 {
        let v: Vec<f32> = (0..4).map(|j| (i * 4 + j) as f32).collect();
        index.insert(v);
    }
    assert_eq!(index.len(), 500);

    let results = index.search(vec![1.0, 0.0, 0.0, 0.0], 10);
    assert_eq!(results.len(), 10);
}
