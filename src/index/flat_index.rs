use crate::index::base::Index;

pub struct FlatIndex {
    pub vectors: Vec<Vec<f32>>,
    pub metadata: Vec<String>,
    pub external_ids: Option<Vec<String>>,
}

impl Index for FlatIndex {
    fn insert(&mut self, vector: Vec<f32>, metadata: String, external_id: Option<String>) {
        self.vectors.push(vector);
        self.metadata.push(metadata);
        if let Some(external_id) = external_id {
            match self.external_ids {
                Some(ref mut external_ids) => external_ids.push(external_id),
                None => self.external_ids = Some(vec![external_id]),
            }
        }
    }

    fn search(&self, query: Vec<f32>, k: usize) -> Vec<SearchResult> {
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

            top_k.push(
                score,
                SearchResult {
                    id: i,
                    score,
                    text: self.metadata[i].clone(),
                    vector: vector.clone(),
                    external_id: match self.external_ids {
                        Some(ref external_ids) => Some(external_ids[i].clone()),
                        None => None,
                    },
                },
            );
        }

        top_k.into_sorted_vec()
    }
}
