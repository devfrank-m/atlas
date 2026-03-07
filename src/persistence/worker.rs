use std::collections::HashSet;

use tokio::sync::{mpsc, oneshot};
use tokio::time::{self, Duration};
use ulid::Ulid;

use crate::api::{AppState, PersistEvent};
use crate::settings::SETTINGS;
use tracing::info;

pub async fn persistence_worker(
    state: AppState,
    mut rx: mpsc::Receiver<PersistEvent>,
    mut shutdown: oneshot::Receiver<()>,
) {
    let mut dirty: HashSet<Ulid> = HashSet::new();
    let mut flush_interval = time::interval(Duration::from_secs(SETTINGS.flush_interval()));

    info!("Persistence worker started");
    loop {
        tokio::select! {
            _ = flush_interval.tick() => flush(&state, &mut dirty),
            evt = rx.recv() => match evt {
                Some(PersistEvent::Dirty(id)) => { dirty.insert(id); }
                Some(PersistEvent::Delete(id)) => {
                    dirty.remove(&id);
                    state.collections.lock().unwrap().remove(&id);
                    if let Err(e) = state.store.delete(&id.to_string()) {
                        tracing::error!("Failed to delete {id}: {e}");
                    }
                }
                None => {
                    flush(&state, &mut dirty);
                    break;
                }
            },
            _ = &mut shutdown => {
                rx.close();
                while let Ok(evt) = rx.try_recv() {
                    match evt {
                        PersistEvent::Dirty(id) => { dirty.insert(id); }
                        PersistEvent::Delete(id) => {
                            dirty.remove(&id);
                            state.collections.lock().unwrap().remove(&id);
                            if let Err(e) = state.store.delete(&id.to_string()) {
                                tracing::error!("Failed to delete {id}: {e}");
                            }
                        }
                    }
                }
                info!("Persistence worker shutting down, flushing {} dirty collection(s)", dirty.len());
                flush(&state, &mut dirty);
                break;
            }
        }
    }
    info!("Persistence worker stopped");
}

fn flush(state: &AppState, dirty: &mut HashSet<Ulid>) {
    if dirty.is_empty() {
        return;
    }
    let ids: Vec<Ulid> = dirty.drain().collect();
    let guard = state.collections.lock().unwrap();
    for id in &ids {
        if let Some(col) = guard.get(id) {
            if let Err(e) = state.store.save(col) {
                tracing::error!("Failed to persist {id}: {e}");
            }
        }
    }
}
