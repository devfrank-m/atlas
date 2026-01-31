mod definitions;
mod similarity;

use definitions::collections::Collection;
use definitions::collections::Metric;

fn main() {
    let vec1 = vec![1.0, 2.0, 3.0];
    let vec2 = vec![7.0, 8.0, 9.0];

    let mut collection = Collection::new("test".to_string(), 3, Metric::Cosine);

    collection.insert(vec1.clone(), "test1".to_string());
    collection.insert(vec2.clone(), "test2".to_string());

    let query = vec![6.0, 7.0, 8.0];
    let results = collection.search(query, 1);
    println!("{:?}", results);
}
