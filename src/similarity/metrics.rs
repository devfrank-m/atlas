pub fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    assert_eq!(
        vec1.len(),
        vec2.len(),
        "Vectors must have the same dimension"
    );

    let mut dot_product = 0.0;
    for i in 0..vec1.len() {
        dot_product += vec1[i] * vec2[i];
    }

    let mut vec1_magnitude = 0.0;
    for i in 0..vec1.len() {
        vec1_magnitude += vec1[i].powi(2);
    }
    vec1_magnitude = vec1_magnitude.sqrt();

    let mut vec2_magnitude = 0.0;
    for i in 0..vec2.len() {
        vec2_magnitude += vec2[i].powi(2);
    }
    vec2_magnitude = vec2_magnitude.sqrt();

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

    let mut dot_product = 0.0;
    for i in 0..vec1.len() {
        dot_product += vec1[i] * vec2[i];
    }
    dot_product
}

pub fn euclidean_distance(vec1: &[f32], vec2: &[f32]) -> f32 {
    assert_eq!(
        vec1.len(),
        vec2.len(),
        "Vectors must have the same dimension"
    );

    let mut sum_of_squares = 0.0;
    for i in 0..vec1.len() {
        sum_of_squares += (vec1[i] - vec2[i]).powi(2);
    }
    sum_of_squares.sqrt()
}
