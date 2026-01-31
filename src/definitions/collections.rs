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

    fn sort_vector_results(&self, results: &mut Vec<SearchResult>) {
        results.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    pub fn search(&self, query: Vec<f32>, k: usize) -> Vec<SearchResult> {

        assert_eq!(query.len(), self.dimension, "Query dimension does not match collection dimension");

        let mut results = Vec::new();
        for (i, vector) in self.vectors.iter().enumerate() {

            // loop through all vectors and calculate the similarity score
            let score = match self.metric {
                Metric::Cosine => {
                    similarity::metrics::cosine_similarity(&query, &vector)
                },
                Metric::DotProduct => {
                    similarity::metrics::dot_product(&query, &vector)
                },
                Metric::Euclidean => {
                    let distance = similarity::metrics::euclidean_distance(&query, &vector);
                    1.0 / (1.0 + distance)
                }
            };

            results.push(SearchResult {
                id: i,
                score,
                text: self.metadata[i].clone(),
                vector: vector.clone(),
            });
        }

        // sort
        self.sort_vector_results(&mut results);


        results.truncate(k);
        results
    }

}
