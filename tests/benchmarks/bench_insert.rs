use atlas::definitions::collections::{Collection, Metric};
use rand::Rng;
use std::time::Instant;

const NUM_VECTORS: usize = 100000;
const DIMENSION: usize = 768;

fn generate_random_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

#[test]
fn bench_insert_10k_vectors() {
    let vectors = generate_random_vectors(NUM_VECTORS, DIMENSION);

    let mut collection = Collection::new("bench".to_string(), DIMENSION, Metric::Cosine);

    let start = Instant::now();
    for (i, v) in vectors.into_iter().enumerate() {
        collection.insert(v, format!("meta-{i}"), None);
    }
    let elapsed = start.elapsed();

    println!("\n--- Insert Benchmark ---");
    println!("Vectors:    {NUM_VECTORS}");
    println!("Dimension:  {DIMENSION}");
    println!("Total time: {elapsed:?}");
    println!("Per insert: {:.2?}", elapsed / NUM_VECTORS as u32);
    println!(
        "Throughput: {:.0} inserts/sec",
        NUM_VECTORS as f64 / elapsed.as_secs_f64()
    );

    assert_eq!(collection.vectors.len(), NUM_VECTORS);
}

#[test]
fn bench_search_after_10k_inserts() {
    let mut rng = rand::thread_rng();
    let vectors = generate_random_vectors(NUM_VECTORS, DIMENSION);

    let mut collection = Collection::new("bench".to_string(), DIMENSION, Metric::Cosine);
    for (i, v) in vectors.into_iter().enumerate() {
        collection.insert(v, format!("meta-{i}"), None);
    }

    let query: Vec<f32> = (0..DIMENSION).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let k = 10;

    let start = Instant::now();
    let results = collection.search(query, k);
    let elapsed = start.elapsed();

    println!("\n--- Search Benchmark (k={k}) ---");
    println!("Vectors:    {NUM_VECTORS}");
    println!("Dimension:  {DIMENSION}");
    println!("Search time: {elapsed:?}");

    assert_eq!(results.len(), k);
}
