use crate::definitions::results::SearchResult;
use crate::similarity;

pub enum Metric {
    Cosine,
    Euclidean,
    DotProduct,
}

pub struct Collection {
    name: String,
    dimension: usize,
    metric: Metric,
    vectors: Vec<Vec<f32>>,
    metadata: Vec<String>,
}

impl Collection {

    pub fn new(name: String, dimension: usize, metric: Metric) -> Self {
        Self {
            name,
            dimension,
            metric,
            vectors: Vec::new(),
            metadata: Vec::new(),
        }
    }

    pub fn insert(&mut self, vector: Vec<f32>, metadata: String) {

        assert_eq!(vector.len(), self.dimension, "Vector dimension does not match collection dimension");

        self.vectors.push(vector);
        self.metadata.push(metadata);
    }

    pub fn search(&self, query: Vec<f32>, k: usize) -> Vec<SearchResult> {

        assert_eq!(query.len(), self.dimension, "Query dimension does not match collection dimension");

        let mut results = Vec::new();
        for (i, vector) in self.vectors.iter().enumerate() {
            let score = match self.metric {
                Metric::Cosine => similarity::metrics::cosine_similarity(&query, &vector),
                Metric::Euclidean => similarity::metrics::euclidean_distance(&query, &vector),
                Metric::DotProduct => similarity::metrics::dot_product(&query, &vector),
            };
            results.push(SearchResult {
                id: i,
                score,
                text: self.metadata[i].clone(),
                vector: vector.clone(),
            });
        }
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(k);
        results
    }

}
