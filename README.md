# Atlas

A lightweight vector similarity search engine with a REST API, built in Rust.

> **Warning**: Not production ready.

## Quickstart

The quickest way to get started is with Docker:

```bash
docker run -p 8600:8600 devfrankm/atlas:latest
```

To persist data across restarts, mount a volume:

```bash
docker run -p 8600:8600 -v atlas-data:/data devfrankm/atlas:latest
```

## API

The server starts on `http://localhost:8600`. Interactive docs are available at `/swagger-ui` and the raw OpenAPI spec at `/api-doc/openapi.json`.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/collections` | Create a collection |
| `GET` | `/collections/{id}` | Get collection details |
| `POST` | `/collections/{id}/vectors` | Insert a vector |
| `POST` | `/collections/{id}/search` | Search for nearest neighbors |

### Example

Create a collection:

```bash
curl -X POST http://localhost:8600/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "my-collection", "dimension": 3, "metric": "cosine", "index": "hnsw"}'
```

Insert a vector:

```bash
curl -X POST http://localhost:8600/collections/{id}/vectors \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3], "metadata": {"label": "foo"}}'
```

Search:

```bash
curl -X POST http://localhost:8600/collections/{id}/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3], "k": 5}'
```

**Supported metrics:** `cosine`, `euclidean`, `dot_product`

**Supported index types:** `flat` (exact), `hnsw` (approximate)

## Run locally

Requires Rust (edition 2024).

```bash
cargo run --release
```
