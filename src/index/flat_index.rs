use crate::definitions::collections::Metric;
use crate::definitions::results::{IndexSearchResult, SearchResult};
use crate::index::base::Index;
use crate::search::heap::TopKHeap;
use crate::similarity;

pub struct FlatIndex {
    pub vectors: Vec<Vec<f32>>,
    pub dimension: usize,
    pub metric: Metric,
}

impl Index for FlatIndex {
    fn insert(&mut self, vector: Vec<f32>) {
        self.vectors.push(vector);
    }

    fn search(&self, query: Vec<f32>, k: usize) -> Vec<IndexSearchResult> {
        assert_eq!(
            query.len(),
            self.dimension,
            "Query dimension does not match collection dimension"
        );

        if k == 0 {
            return Vec::new();
        }

        let mut top_k = TopKHeap::new(k);

        for (i, vector) in self.vectors.iter().enumerate() {
            let score = match self.metric {
                Metric::Cosine => similarity::metrics::cosine_similarity(&query, &vector),
                Metric::DotProduct => similarity::metrics::dot_product(&query, &vector),
                Metric::Euclidean => {
                    let distance = similarity::metrics::euclidean_distance(&query, &vector);
                    1.0 / (1.0 + distance)
                }
            };

            top_k.push(score, IndexSearchResult { id: i, score });
        }

        top_k.into_sorted_vec()
    }
}
