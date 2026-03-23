use crate::definitions::metadata::Metadata;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Filter {
    Eq {
        key: String,
        #[schema(value_type = Object)]
        value: Value,
    },
}

impl Filter {
    pub fn matches(&self, metadata: &Metadata) -> bool {
        match self {
            Filter::Eq { key, value } => metadata.get(key) == Some(value),
        }
    }
}
