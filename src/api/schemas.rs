use serde::{Deserialize, Serialize};


#[derive(Serialize, Deserialize)]
pub struct CollectionCreateRequest {
    pub name: String,
    pub dimension: usize,
    pub metric: String,
}

#[derive(Serialize, Deserialize)]
pub struct CollectionCreateResponse {
    pub id: String,
}
