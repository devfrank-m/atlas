use crate::definitions::results::IndexSearchResult;

pub trait Index: Send {
    fn insert(&mut self, vector: Vec<f32>);
    fn search(&self, query: Vec<f32>, k: usize) -> Vec<IndexSearchResult>;
    fn get(&self, id: usize) -> Option<&Vec<f32>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
