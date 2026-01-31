pub fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    assert_eq!(
        vec1.len(),
        vec2.len(),
        "Vectors must have the same dimension"
    );

    let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();

    let vec1_magnitude: f32 = vec1.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let vec2_magnitude: f32 = vec2.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

    if vec1_magnitude == 0.0 || vec2_magnitude == 0.0 {
        return 0.0;
    }

    dot_product / (vec1_magnitude * vec2_magnitude)
}

pub fn dot_product(vec1: &[f32], vec2: &[f32]) -> f32 {
    assert_eq!(
        vec1.len(),
        vec2.len(),
        "Vectors must have the same dimension"
    );

    vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum()
}

pub fn euclidean_distance(vec1: &[f32], vec2: &[f32]) -> f32 {
    assert_eq!(
        vec1.len(),
        vec2.len(),
        "Vectors must have the same dimension"
    );

    vec1.iter()
        .zip(vec2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}
