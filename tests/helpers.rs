use serde_json::json;
use std::collections::HashMap;

pub fn meta(label: &str) -> HashMap<String, serde_json::Value> {
    HashMap::from([("label".to_string(), json!(label))])
}
