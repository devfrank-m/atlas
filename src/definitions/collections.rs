use crate::definitions::filter::Filter;
use crate::definitions::metadata::Metadata;
use crate::definitions::results::SearchResult;
use crate::index::base::Index;
use crate::index::flat::FlatIndex;
use crate::index::hnsw::HnswIndex;
use ulid::Ulid;

#[derive(Clone, Copy)]
pub enum Metric {
    Cosine,
    Euclidean,
    DotProduct,
}

#[derive(Clone, Copy)]
pub enum IndexType {
    Flat,
    Hnsw,
}

pub struct Collection {
    pub id: Ulid,
    pub name: String,
    pub dimension: usize,
    pub metric: Metric,
    pub index: Box<dyn Index>,
    pub metadata: Vec<Metadata>,
    pub external_ids: Option<Vec<String>>,
}

impl Collection {
    pub fn new(name: String, dimension: usize, metric: Metric, index_type: IndexType) -> Self {
        let index: Box<dyn Index> = match index_type {
            IndexType::Flat => Box::new(FlatIndex {
                vectors: Vec::new(),
                dimension,
                metric,
            }),
            IndexType::Hnsw => Box::new(
                // intentionally hard-coded parameters for now,
                // we'll make them configurable on API once we
                // narrow down on the most ergonomic schema for index configuration
                HnswIndex::new(dimension, metric, 16, 200),
            ),
        };

        Self {
            id: Ulid::new(),
            name,
            dimension,
            index,
            metric,
            metadata: Vec::new(),
            external_ids: None,
        }
    }

    pub fn insert(&mut self, vector: Vec<f32>, metadata: Metadata, external_id: Option<String>) {
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

    pub fn search(&self, query: Vec<f32>, k: usize, filter: Option<Filter>) -> Vec<SearchResult> {
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

        let search_k = if filter.clone().is_some() {
            // also hard coding for now, should move this to config
            let factor = 5;
            let cap = 2000;
            (k * factor).min(cap).min(self.index.len())
        } else {
            k
        };

        let index_results = self.index.search(query, search_k);
        let mut search_results = Vec::with_capacity(index_results.len());

        for index_result in index_results {
            if search_results.len() == k {
                // if we've already collected k results,
                // we can stop processing further results
                break;
            }

            let metadata = self
                .metadata
                .get(index_result.id)
                .cloned()
                .unwrap_or_default();

            if let Some(filter_value) = &filter {
                if !filter_value.matches(&metadata) {
                    // if the result doesn't match the filter,
                    // we skip it and continue to the next one
                    continue;
                }
            }

            let external_id = self
                .external_ids
                .as_ref()
                .and_then(|ids| ids.get(index_result.id))
                .cloned()
                .filter(|s| !s.is_empty());

            let vector = self.index.get(index_result.id).map(|v| v.to_vec());
            search_results.push(SearchResult {
                id: index_result.id,
                score: index_result.score,
                metadata,
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
