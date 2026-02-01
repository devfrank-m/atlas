pub mod collections;
pub mod schemas;

use atlas::definitions::collections::Collection;
use std::sync::{Arc, Mutex};

pub type AppState = Arc<AppStateInner>;

pub struct AppStateInner {
    pub collections: Mutex<Vec<Collection>>,
}

impl AppStateInner {
    pub fn new() -> Self {
        Self {
            collections: Mutex::new(Vec::new()),
        }
    }
}
