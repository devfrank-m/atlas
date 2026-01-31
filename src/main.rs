mod similarity;
use similarity::metrics;

fn main() {
    let vec1 = vec![1.0, 2.0, 3.0];
    let vec2 = vec![1.0, 3.0, 3.0];

    let similarity = metrics::cosine_similarity(&vec1, &vec2);
    println!("Similarity: {}", similarity);

    let dot_product = metrics::dot_product(&vec1, &vec2);
    println!("Dot Product: {}", dot_product);

    let euclidean_distance = metrics::euclidean_distance(&vec1, &vec2);
    println!("Euclidean Distance: {}", euclidean_distance);
}
