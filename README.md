# Atlas

A lightweight vector similarity search engine with a REST API, built in Rust.

> **Warning**: Not production ready.

## Run locally

Requires Rust (edition 2024).

```bash
cargo run --release
```

Server starts on `http://0.0.0.0:8600`.

## Run with Docker

```bash
docker run -p 8600:8600 devfrankm/atlas:latest
```

Or pin to a specific version:

```bash
docker run -p 8600:8600 devfrankm/atlas:0.0.1
```
