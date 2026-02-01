// mod definitions;
// mod similarity;

// use definitions::collections::Collection;
// use definitions::collections::Metric;
mod api;
use api::collections::create_collection;

use axum::{
    Router,
    routing::{get, post},
};
use tracing::info;

// fn main() {
//     let vec1 = vec![1.0, 2.0, 3.0];
//     let vec2 = vec![7.0, 8.0, 9.0];

//     let mut collection = Collection::new("test".to_string(), 3, Metric::Cosine);

//     collection.insert(vec1.clone(), "test1".to_string());
//     collection.insert(vec2.clone(), "test2".to_string());

//     let query = vec![6.0, 7.0, 8.0];
//     let results = collection.search(query, 1);
//     println!("{:?}", results);
// }

async fn get_collection() -> String {
    "Hello, World!".to_string()
}

async fn insert_vector() -> String {
    "Hello, World!".to_string()
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app = Router::new()
        .route("/collections", post(create_collection))
        .route("/collections/{id}", get(get_collection))
        .route("/collections/{id}/vectors", post(insert_vector));

    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8600").await.unwrap();
    info!("Server running on http://0.0.0.0:8600");
    axum::serve(listener, app).await.unwrap();
}
