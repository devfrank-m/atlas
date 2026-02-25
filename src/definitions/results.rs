use crate::definitions::metadata::Metadata;
use std::fmt::{Debug, Formatter};

pub struct SearchResult {
    pub id: usize,
    pub score: f32,
    pub metadata: Metadata,
    pub vector: Option<Vec<f32>>,
    pub external_id: Option<String>,
}

pub struct IndexSearchResult {
    pub id: usize,
    pub score: f32,
}

impl Debug for SearchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchResult")
            .field("id", &self.id)
            .field("score", &self.score)
            .field("metadata", &self.metadata)
            .field("vector", &self.vector)
            .field("external_id", &self.external_id)
            .finish()
    }
}

impl Debug for IndexSearchResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexSearchResult")
            .field("id", &self.id)
            .field("score", &self.score)
            .finish()
    }
}
