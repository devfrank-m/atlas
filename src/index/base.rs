trait Index {
    fn insert(&mut self, vector: Vec<f32>, metadata: String, external_id: Option<String>);
    fn search(&self, query: Vec<f32>, k: usize) -> Vec<SearchResult>;
}
