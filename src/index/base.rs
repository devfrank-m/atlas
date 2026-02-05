use crate::definitions::results::{IndexSearchResult, SearchResult};

pub trait Index {
    fn insert(&mut self, vector: Vec<f32>);
    fn search(&self, query: Vec<f32>, k: usize) -> Vec<IndexSearchResult>;
}
