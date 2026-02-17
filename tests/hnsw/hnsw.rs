use atlas::definitions::collections::Metric;
use atlas::index::base::Index;
use atlas::index::flat_index::FlatIndex;
use atlas::index::hnsw::HnswIndex;
use std::collections::HashSet;

fn new_hnsw(dim: usize, metric: Metric) -> HnswIndex {
    HnswIndex::new(dim, metric, 16, 200)
}

#[test]
fn test_empty_index() {
    let index = new_hnsw(3, Metric::Cosine);
    let results = index.search(vec![1.0, 0.0, 0.0], 10);
    assert_eq!(results.len(), 0);
}

#[test]
fn test_k_zero_returns_empty() {
    let mut index = new_hnsw(3, Metric::Cosine);
    index.insert(vec![1.0, 0.0, 0.0]);
    let results = index.search(vec![1.0, 0.0, 0.0], 0);
    assert_eq!(results.len(), 0);
}

#[test]
fn test_single_vector() {
    let mut index = new_hnsw(3, Metric::Cosine);
    index.insert(vec![1.0, 0.0, 0.0]);

    let results = index.search(vec![1.0, 0.0, 0.0], 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 0);
    assert!((results[0].score - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_search() {
    let mut index = new_hnsw(3, Metric::Cosine);
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
    let mut index = new_hnsw(2, Metric::DotProduct);
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
    let mut index = new_hnsw(2, Metric::Euclidean);
    index.insert(vec![1.0, 0.0]);
    index.insert(vec![10.0, 10.0]);

    let results = index.search(vec![1.0, 0.0], 2);

    assert_eq!(results[0].id, 0);
    assert!((results[0].score - 1.0).abs() < 1e-6);
    assert!(results[1].score < results[0].score);
}

#[test]
fn test_results_sorted_descending() {
    let mut index = new_hnsw(2, Metric::Cosine);
    index.insert(vec![0.0, 1.0]);
    index.insert(vec![1.0, 0.0]);
    index.insert(vec![1.0, 1.0]);

    let results = index.search(vec![1.0, 0.0], 3);

    for i in 0..results.len() - 1 {
        assert!(
            results[i].score >= results[i + 1].score,
            "results should be sorted by score descending"
        );
    }
}

#[test]
fn test_k_larger_than_index() {
    let mut index = new_hnsw(2, Metric::Cosine);
    index.insert(vec![1.0, 0.0]);
    index.insert(vec![0.0, 1.0]);

    let results = index.search(vec![1.0, 0.0], 10);
    assert_eq!(results.len(), 2);
}

#[test]
fn test_get_returns_correct_vector() {
    let mut index = new_hnsw(3, Metric::Cosine);
    let v = vec![1.0, 2.0, 3.0];
    index.insert(v.clone());

    assert_eq!(index.get(0), Some(v.as_slice()));
    assert_eq!(index.get(1), None);
}

#[test]
fn test_len_and_is_empty() {
    let mut index = new_hnsw(2, Metric::Cosine);
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);

    index.insert(vec![1.0, 0.0]);
    assert!(!index.is_empty());
    assert_eq!(index.len(), 1);

    index.insert(vec![0.0, 1.0]);
    assert_eq!(index.len(), 2);
}

#[test]
#[should_panic(expected = "Vector dimension does not match")]
fn test_insert_wrong_dimension_panics() {
    let mut index = new_hnsw(3, Metric::Cosine);
    index.insert(vec![1.0, 2.0]);
}

#[test]
#[should_panic(expected = "Query dimension does not match")]
fn test_search_wrong_dimension_panics() {
    let index = new_hnsw(3, Metric::Cosine);
    index.search(vec![1.0, 2.0], 1);
}

#[test]
fn test_many_inserts() {
    let mut index = new_hnsw(4, Metric::Cosine);
    for i in 0..500 {
        let v: Vec<f32> = (0..4).map(|j| (i * 4 + j) as f32).collect();
        index.insert(v);
    }
    assert_eq!(index.len(), 500);

    let results = index.search(vec![1.0, 0.0, 0.0, 0.0], 10);
    assert_eq!(results.len(), 10);
}

#[test]
fn test_recall_against_flat_index() {
    use rand::RngExt;

    let dim = 32;
    let n = 1000;
    let k = 10;
    let num_queries = 20;
    let mut rng = rand::rng();

    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect())
        .collect();

    let mut flat = FlatIndex {
        vectors: Vec::new(),
        dimension: dim,
        metric: Metric::Cosine,
    };
    let mut hnsw = HnswIndex::new(dim, Metric::Cosine, 16, 200);
    hnsw.set_ef_search(100);

    for v in &vectors {
        flat.insert(v.clone());
        hnsw.insert(v.clone());
    }

    let mut total_recall = 0.0;
    for _ in 0..num_queries {
        let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect();

        let flat_ids: HashSet<usize> = flat.search(query.clone(), k).iter().map(|r| r.id).collect();
        let hnsw_ids: HashSet<usize> = hnsw.search(query, k).iter().map(|r| r.id).collect();

        let intersection = flat_ids.intersection(&hnsw_ids).count();
        total_recall += intersection as f64 / k as f64;
    }

    let avg_recall = total_recall / num_queries as f64;
    assert!(
        avg_recall > 0.8,
        "recall@{k} should be above 0.8, got {avg_recall:.4}"
    );
}

#[test]
fn test_higher_ef_search_improves_recall() {
    use rand::RngExt;

    let dim = 32;
    let n = 500;
    let k = 10;
    let num_queries = 20;
    let mut rng = rand::rng();

    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect())
        .collect();

    let mut flat = FlatIndex {
        vectors: Vec::new(),
        dimension: dim,
        metric: Metric::Cosine,
    };
    let mut hnsw = HnswIndex::new(dim, Metric::Cosine, 16, 200);

    for v in &vectors {
        flat.insert(v.clone());
        hnsw.insert(v.clone());
    }

    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect())
        .collect();

    let compute_recall = |hnsw: &HnswIndex, flat: &FlatIndex| -> f64 {
        let mut total = 0.0;
        for q in &queries {
            let flat_ids: HashSet<usize> = flat.search(q.clone(), k).iter().map(|r| r.id).collect();
            let hnsw_ids: HashSet<usize> = hnsw.search(q.clone(), k).iter().map(|r| r.id).collect();
            total += flat_ids.intersection(&hnsw_ids).count() as f64 / k as f64;
        }
        total / num_queries as f64
    };

    hnsw.set_ef_search(10);
    let recall_low = compute_recall(&hnsw, &flat);

    hnsw.set_ef_search(200);
    let recall_high = compute_recall(&hnsw, &flat);

    assert!(
        recall_high >= recall_low,
        "higher ef_search should give equal or better recall: low={recall_low:.4} high={recall_high:.4}"
    );
}
