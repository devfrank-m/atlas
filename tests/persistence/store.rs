use std::collections::HashMap;
use std::path::PathBuf;

use atlas::definitions::collections::{Collection, IndexType, Metric};
use atlas::persistence::{CollectionStore, WalRecord};
use ulid::Ulid;

fn temp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("atlas_test_{}", Ulid::new()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn empty_meta() -> HashMap<String, serde_json::Value> {
    HashMap::new()
}

// ---------------------------------------------------------------------------
// Flat index — save / load roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_save_and_load_flat() {
    let dir = temp_dir();
    let store = CollectionStore::new(&dir).unwrap();

    let mut col = Collection::new("planets".to_string(), 3, Metric::Cosine, IndexType::Flat);
    col.insert(vec![1.0, 0.0, 0.0], empty_meta(), None);
    col.insert(vec![0.0, 1.0, 0.0], empty_meta(), Some("ext-1".to_string()));

    store.save(&col).unwrap();

    let loaded = store.load(&col.id.to_string()).unwrap();

    assert_eq!(loaded.name, "planets");
    assert_eq!(loaded.dimension, 3);
    assert_eq!(loaded.index.len(), 2);
    assert!(loaded.external_ids.is_some());
}

#[test]
fn test_flat_search_after_load() {
    let dir = temp_dir();
    let store = CollectionStore::new(&dir).unwrap();

    let mut col = Collection::new("test".to_string(), 2, Metric::Cosine, IndexType::Flat);
    col.insert(vec![1.0, 0.0], empty_meta(), None);
    col.insert(vec![0.0, 1.0], empty_meta(), None);

    store.save(&col).unwrap();
    let loaded = store.load(&col.id.to_string()).unwrap();

    let results = loaded.search(vec![1.0, 0.0], 1, None);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 0);
}

// ---------------------------------------------------------------------------
// HNSW index — save / load roundtrip (uses .vec file + mmap)
// ---------------------------------------------------------------------------

#[test]
fn test_save_and_load_hnsw() {
    let dir = temp_dir();
    let store = CollectionStore::new(&dir).unwrap();

    let mut col = Collection::new("stars".to_string(), 4, Metric::Euclidean, IndexType::Hnsw);
    col.insert(vec![1.0, 0.0, 0.0, 0.0], empty_meta(), None);
    col.insert(vec![0.0, 1.0, 0.0, 0.0], empty_meta(), None);
    col.insert(vec![0.0, 0.0, 1.0, 0.0], empty_meta(), None);

    store.save(&col).unwrap();

    assert!(dir.join(format!("{}.1.vec", col.id)).exists());

    let loaded = store.load(&col.id.to_string()).unwrap();

    assert_eq!(loaded.name, "stars");
    assert_eq!(loaded.dimension, 4);
    assert_eq!(loaded.index.len(), 3);
}

#[test]
fn test_hnsw_search_after_load() {
    let dir = temp_dir();
    let store = CollectionStore::new(&dir).unwrap();

    let mut col = Collection::new("test".to_string(), 2, Metric::DotProduct, IndexType::Hnsw);
    col.insert(vec![1.0, 0.0], empty_meta(), None);
    col.insert(vec![0.0, 1.0], empty_meta(), None);

    store.save(&col).unwrap();
    let loaded = store.load(&col.id.to_string()).unwrap();

    let results = loaded.search(vec![1.0, 0.0], 1, None);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 0);
}

#[test]
fn test_hnsw_insert_after_load_converts_mmap_to_owned() {
    let dir = temp_dir();
    let store = CollectionStore::new(&dir).unwrap();

    let mut col = Collection::new("test".to_string(), 2, Metric::Cosine, IndexType::Hnsw);
    col.insert(vec![1.0, 0.0], empty_meta(), None);
    store.save(&col).unwrap();

    let mut loaded = store.load(&col.id.to_string()).unwrap();
    // inserting into a mmap-backed HNSW should trigger ensure_owned
    loaded.insert(vec![0.0, 1.0], empty_meta(), None);
    assert_eq!(loaded.index.len(), 2);
}

// ---------------------------------------------------------------------------
// Generation counter
// ---------------------------------------------------------------------------

#[test]
fn test_generation_increments_on_each_save() {
    use atlas::persistence::CollectionSnapshot;

    let dir = temp_dir();
    let store = CollectionStore::new(&dir).unwrap();
    let col = Collection::new("test".to_string(), 2, Metric::Cosine, IndexType::Flat);
    let id = col.id.to_string();

    store.save(&col).unwrap();
    let snap1: CollectionSnapshot =
        serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{id}.json"))).unwrap())
            .unwrap();
    assert_eq!(snap1.header.generation, 1);

    store.save(&col).unwrap();
    let snap2: CollectionSnapshot =
        serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{id}.json"))).unwrap())
            .unwrap();
    assert_eq!(snap2.header.generation, 2);
}

// ---------------------------------------------------------------------------
// Checksum verification
// ---------------------------------------------------------------------------

#[test]
fn test_corrupted_file_fails_to_load() {
    let dir = temp_dir();
    let store = CollectionStore::new(&dir).unwrap();

    let col = Collection::new("test".to_string(), 2, Metric::Cosine, IndexType::Flat);
    let id = col.id.to_string();
    store.save(&col).unwrap();

    let path = dir.join(format!("{id}.json"));
    let mut json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    json["header"]["checksum"] = serde_json::Value::String("badhash".to_string());
    std::fs::write(&path, serde_json::to_string(&json).unwrap()).unwrap();

    let result = store.load(&id);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("checksum"));
}

// ---------------------------------------------------------------------------
// load_all
// ---------------------------------------------------------------------------

#[test]
fn test_load_all_returns_all_collections() {
    let dir = temp_dir();
    let store = CollectionStore::new(&dir).unwrap();

    let col1 = Collection::new("a".to_string(), 2, Metric::Cosine, IndexType::Flat);
    let col2 = Collection::new("b".to_string(), 3, Metric::Euclidean, IndexType::Flat);
    store.save(&col1).unwrap();
    store.save(&col2).unwrap();

    let all = store.load_all().unwrap();
    assert_eq!(all.len(), 2);
}

#[test]
fn test_load_all_skips_corrupted_files() {
    let dir = temp_dir();
    let store = CollectionStore::new(&dir).unwrap();

    let col = Collection::new("good".to_string(), 2, Metric::Cosine, IndexType::Flat);
    store.save(&col).unwrap();

    std::fs::write(dir.join("bad.json"), b"not valid json").unwrap();

    let all = store.load_all().unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].name, "good");
}

// ---------------------------------------------------------------------------
// delete
// ---------------------------------------------------------------------------

#[test]
fn test_delete_removes_json_and_vec_files() {
    let dir = temp_dir();
    let store = CollectionStore::new(&dir).unwrap();

    let mut col = Collection::new("test".to_string(), 2, Metric::Cosine, IndexType::Hnsw);
    col.insert(vec![1.0, 0.0], empty_meta(), None);
    let id = col.id.to_string();
    store.save(&col).unwrap();

    assert!(dir.join(format!("{id}.json")).exists());
    assert!(dir.join(format!("{id}.1.vec")).exists());

    store.delete(&id).unwrap();

    assert!(!dir.join(format!("{id}.json")).exists());
    assert!(!dir.join(format!("{id}.1.vec")).exists());
}

#[test]
fn test_delete_nonexistent_is_ok() {
    let dir = temp_dir();
    let store = CollectionStore::new(&dir).unwrap();
    assert!(store.delete("00000000000000000000000000").is_ok());
}

// ---------------------------------------------------------------------------
// WAL — append / replay
// ---------------------------------------------------------------------------

#[test]
fn test_wal_replay_on_load() {
    let dir = temp_dir();
    let store = CollectionStore::new(&dir).unwrap();

    let col = Collection::new("test".to_string(), 2, Metric::Cosine, IndexType::Flat);
    let id = col.id.to_string();
    store.save(&col).unwrap();

    // Append WAL records directly (simulating inserts that happened after last snapshot)
    store
        .wal_append(
            &id,
            &WalRecord::Insert {
                vector: vec![1.0, 0.0],
                metadata: empty_meta(),
                external_id: None,
            },
        )
        .unwrap();
    store
        .wal_append(
            &id,
            &WalRecord::Insert {
                vector: vec![0.0, 1.0],
                metadata: empty_meta(),
                external_id: Some("ext".to_string()),
            },
        )
        .unwrap();

    assert!(dir.join(format!("{id}.wal")).exists());

    let loaded = store.load(&id).unwrap();

    assert_eq!(loaded.index.len(), 2);
    // WAL file is truncated after successful replay
    assert!(!dir.join(format!("{id}.wal")).exists());
}

#[test]
fn test_wal_append_multiple_records() {
    let dir = temp_dir();
    let store = CollectionStore::new(&dir).unwrap();

    let col = Collection::new("test".to_string(), 2, Metric::Cosine, IndexType::Flat);
    let id = col.id.to_string();

    for i in 0..5 {
        store
            .wal_append(
                &id,
                &WalRecord::Insert {
                    vector: vec![i as f32, 0.0],
                    metadata: empty_meta(),
                    external_id: None,
                },
            )
            .unwrap();
    }

    let wal_content = std::fs::read_to_string(dir.join(format!("{id}.wal"))).unwrap();
    assert_eq!(wal_content.lines().count(), 5);
}

// ---------------------------------------------------------------------------
// Snapshot header fields
// ---------------------------------------------------------------------------

#[test]
fn test_snapshot_header_fields_flat() {
    use atlas::persistence::CollectionSnapshot;

    let dir = temp_dir();
    let store = CollectionStore::new(&dir).unwrap();

    let mut col = Collection::new("test".to_string(), 3, Metric::Cosine, IndexType::Flat);
    col.insert(vec![1.0, 0.0, 0.0], empty_meta(), None);
    col.insert(vec![0.0, 1.0, 0.0], empty_meta(), None);
    let id = col.id.to_string();
    store.save(&col).unwrap();

    let snap: CollectionSnapshot =
        serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{id}.json"))).unwrap())
            .unwrap();

    assert_eq!(snap.header.version, 2);
    assert_eq!(snap.header.index_type, "flat");
    assert_eq!(snap.header.vector_count, 2);
    assert_eq!(snap.header.dimension, 3);
}

#[test]
fn test_snapshot_header_fields_hnsw() {
    use atlas::persistence::CollectionSnapshot;

    let dir = temp_dir();
    let store = CollectionStore::new(&dir).unwrap();

    let mut col = Collection::new("test".to_string(), 2, Metric::Euclidean, IndexType::Hnsw);
    col.insert(vec![1.0, 0.0], empty_meta(), None);
    col.insert(vec![0.0, 1.0], empty_meta(), None);
    col.insert(vec![1.0, 1.0], empty_meta(), None);
    let id = col.id.to_string();
    store.save(&col).unwrap();

    let snap: CollectionSnapshot =
        serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{id}.json"))).unwrap())
            .unwrap();

    assert_eq!(snap.header.index_type, "hnsw");
    assert_eq!(snap.header.vector_count, 3);
    assert_eq!(snap.header.dimension, 2);
}

// ---------------------------------------------------------------------------
// Collection snapshot roundtrip (to_snapshot / from_snapshot)
// ---------------------------------------------------------------------------

#[test]
fn test_collection_snapshot_roundtrip_flat() {
    let dir = temp_dir();
    let store = CollectionStore::new(&dir).unwrap();

    let mut original = Collection::new(
        "roundtrip".to_string(),
        3,
        Metric::DotProduct,
        IndexType::Flat,
    );
    original.insert(vec![1.0, 2.0, 3.0], empty_meta(), Some("a".to_string()));
    original.insert(vec![4.0, 5.0, 6.0], empty_meta(), None);

    store.save(&original).unwrap();
    let restored = store.load(&original.id.to_string()).unwrap();

    assert_eq!(restored.id, original.id);
    assert_eq!(restored.name, original.name);
    assert_eq!(restored.dimension, original.dimension);
    assert_eq!(restored.index.len(), original.index.len());

    let orig_v0 = original.index.get(0).unwrap().to_vec();
    let rest_v0 = restored.index.get(0).unwrap().to_vec();
    assert_eq!(orig_v0, rest_v0);
}

#[test]
fn test_collection_snapshot_roundtrip_hnsw() {
    let dir = temp_dir();
    let store = CollectionStore::new(&dir).unwrap();

    let mut original = Collection::new("roundtrip".to_string(), 4, Metric::Cosine, IndexType::Hnsw);
    original.insert(vec![1.0, 0.0, 0.0, 0.0], empty_meta(), None);
    original.insert(vec![0.0, 1.0, 0.0, 0.0], empty_meta(), None);
    original.insert(vec![0.0, 0.0, 1.0, 0.0], empty_meta(), None);

    store.save(&original).unwrap();
    let restored = store.load(&original.id.to_string()).unwrap();

    assert_eq!(restored.id, original.id);
    assert_eq!(restored.index.len(), 3);

    let orig_v1 = original.index.get(1).unwrap().to_vec();
    let rest_v1 = restored.index.get(1).unwrap().to_vec();
    assert_eq!(orig_v1, rest_v1);
}
