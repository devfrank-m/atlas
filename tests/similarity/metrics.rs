use atlas::similarity::metrics::*;

#[test]
fn test_cosine_similarity_identical_vectors() {
    let vec1 = vec![1.0, 2.0, 3.0];
    let vec2 = vec![1.0, 2.0, 3.0];
    let result = cosine_similarity(&vec1, &vec2);
    assert!((result - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_orthogonal_vectors() {
    let vec1 = vec![1.0, 0.0];
    let vec2 = vec![0.0, 1.0];
    let result = cosine_similarity(&vec1, &vec2);
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_opposite_vectors() {
    let vec1 = vec![1.0, 2.0, 3.0];
    let vec2 = vec![-1.0, -2.0, -3.0];
    let result = cosine_similarity(&vec1, &vec2);
    assert!((result - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_partial_match() {
    let vec1 = vec![1.0, 2.0, 3.0];
    let vec2 = vec![2.0, 4.0, 6.0];
    let result = cosine_similarity(&vec1, &vec2);
    assert!((result - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_zero_vector() {
    let vec1 = vec![0.0, 0.0, 0.0];
    let vec2 = vec![1.0, 2.0, 3.0];
    let result = cosine_similarity(&vec1, &vec2);
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
#[should_panic]
fn test_cosine_similarity_different_lengths() {
    let vec1 = vec![1.0, 2.0];
    let vec2 = vec![1.0, 2.0, 3.0];
    cosine_similarity(&vec1, &vec2);
}

#[test]
fn test_dot_product_basic() {
    let vec1 = vec![1.0, 2.0, 3.0];
    let vec2 = vec![4.0, 5.0, 6.0];
    let result = dot_product(&vec1, &vec2);
    assert!((result - 32.0).abs() < 1e-6);
}

#[test]
fn test_dot_product_zero_vector() {
    let vec1 = vec![0.0, 0.0, 0.0];
    let vec2 = vec![1.0, 2.0, 3.0];
    let result = dot_product(&vec1, &vec2);
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn test_dot_product_negative_values() {
    let vec1 = vec![-1.0, 2.0, -3.0];
    let vec2 = vec![1.0, -2.0, 3.0];
    let result = dot_product(&vec1, &vec2);
    assert!((result - (-14.0)).abs() < 1e-6);
}

#[test]
#[should_panic]
fn test_dot_product_different_lengths() {
    let vec1 = vec![1.0, 2.0];
    let vec2 = vec![1.0, 2.0, 3.0];
    dot_product(&vec1, &vec2);
}

#[test]
fn test_euclidean_distance_identical_vectors() {
    let vec1 = vec![1.0, 2.0, 3.0];
    let vec2 = vec![1.0, 2.0, 3.0];
    let result = euclidean_distance(&vec1, &vec2);
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn test_euclidean_distance_basic() {
    let vec1 = vec![1.0, 2.0, 3.0];
    let vec2 = vec![4.0, 6.0, 8.0];
    let result = euclidean_distance(&vec1, &vec2);
    assert!((result - 7.0710678118654755).abs() < 1e-6);
}

#[test]
fn test_euclidean_distance_negative_values() {
    let vec1 = vec![-1.0, -2.0, -3.0];
    let vec2 = vec![1.0, 2.0, 3.0];
    let result = euclidean_distance(&vec1, &vec2);
    assert!((result - 7.483314773547883).abs() < 1e-6);
}

#[test]
fn test_euclidean_distance_one_dimension() {
    let vec1 = vec![5.0];
    let vec2 = vec![10.0];
    let result = euclidean_distance(&vec1, &vec2);
    assert!((result - 5.0).abs() < 1e-6);
}

#[test]
#[should_panic]
fn test_euclidean_distance_different_lengths() {
    let vec1 = vec![1.0, 2.0];
    let vec2 = vec![1.0, 2.0, 3.0];
    euclidean_distance(&vec1, &vec2);
}
