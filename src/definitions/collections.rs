use crate::definitions::results::SearchResult;
use crate::index::base::Index;
use crate::index::flat_index::FlatIndex;
use ulid::Ulid;

#[derive(Clone)]
pub enum Metric {
    Cosine,
    Euclidean,
    DotProduct,
}

pub struct Collection {
    pub id: Ulid,
    pub name: String,
    pub dimension: usize,
    pub metric: Metric,
    pub index: Box<dyn Index>,
    pub metadata: Vec<String>,
    pub external_ids: Option<Vec<String>>,
}

impl Collection {
    pub fn new(name: String, dimension: usize, metric: Metric) -> Self {
        Self {
            id: Ulid::new(),
            name,
            dimension,
            index: Box::new(FlatIndex {
                vectors: Vec::new(),
                dimension,
                metric: metric.clone(),
            }),
            metric,
            metadata: Vec::new(),
            external_ids: None,
        }
    }

    pub fn insert(&mut self, vector: Vec<f32>, metadata: String, external_id: Option<String>) {
        assert_eq!(
            vector.len(),
            self.dimension,
            "Vector dimension does not match collection dimension"
        );

        self.index.insert(vector);
        self.metadata.push(metadata);

        // not very readable, is it?
        match external_id {
            Some(external_id) => match self.external_ids {
                Some(ref mut external_ids) => external_ids.push(external_id),
                None => self.external_ids = Some(vec![external_id]),
            },
            None => {
                // TODO: we should probably have a better way to handle this,
                // but for now we just push an empty string to keep the indices aligned
                match self.external_ids {
                    Some(ref mut external_ids) => external_ids.push(String::new()),
                    None => self.external_ids = Some(vec![String::new()]),
                }
            }
        }
    }

    pub fn search(&self, query: Vec<f32>, k: usize) -> Vec<SearchResult> {
        assert_eq!(
            query.len(),
            self.dimension,
            "Query dimension does not match collection dimension"
        );

        if k == 0 {
            return Vec::new();
        }

        if self.index.is_empty() {
            return Vec::new();
        }
        
        let index_results = self.index.search(query, k);
        let mut search_results = Vec::with_capacity(index_results.len());

        for index_result in index_results {
            let metadata = self
                .metadata
                .get(index_result.id)
                .cloned()
                .unwrap_or_default();
            let external_id = self
                .external_ids
                .as_ref()
                .and_then(|ids| ids.get(index_result.id))
                .cloned()
                .filter(|s| !s.is_empty());
            let vector = self.index.get(index_result.id).cloned();
            search_results.push(SearchResult {
                id: index_result.id,
                score: index_result.score,
                text: metadata,
                vector,
                external_id,
            });
        }

        search_results
    }
}

impl std::fmt::Debug for Collection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Collection")
            .field("id", &self.id.to_string())
            .field("name", &self.name)
            .field("dimension", &self.dimension)
            .finish()
    }
}
