use std::fmt::Debug;

pub struct SearchResult {
    pub id: usize,
    pub score: f32,
    pub text: String,
    pub vector: Vec<f32>,
}


impl Debug for SearchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchResult")
            .field("id", &self.id)
            .field("score", &self.score)
            .field("text", &self.text)
            .field("vector", &self.vector)
            .finish()
    }
}