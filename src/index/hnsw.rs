use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use rand::RngExt;
use serde::{Deserialize, Serialize};

use crate::definitions::collections::Metric;
use crate::definitions::results::IndexSearchResult;
use crate::index::base::Index;
use crate::persistence::{IndexSnapshot, NeighborSnapshot, NodeSnapshot, VectorStorage};
use crate::similarity;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct Neighbor {
    id: usize,
    distance: f32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Neighbor {}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Node {
    connections: Vec<Vec<Neighbor>>,
}

struct SearchScratch {
    visited: Vec<u32>,
    visited_gen: u32,
    candidates: BinaryHeap<Reverse<Neighbor>>,
    results: BinaryHeap<Neighbor>,
}

impl SearchScratch {
    fn new() -> Self {
        Self {
            visited: Vec::new(),
            visited_gen: 0,
            candidates: BinaryHeap::new(),
            results: BinaryHeap::new(),
        }
    }

    fn prepare(&mut self, capacity: usize) {
        if self.visited.len() < capacity {
            self.visited.resize(capacity, 0);
        }
        self.visited_gen = self.visited_gen.wrapping_add(1);
        if self.visited_gen == 0 {
            self.visited.fill(0);
            self.visited_gen = 1;
        }
        self.candidates.clear();
        self.results.clear();
    }
}

#[inline]
fn get_vector(vectors: &[f32], dimension: usize, id: usize) -> &[f32] {
    let start = id * dimension;
    &vectors[start..start + dimension]
}

#[inline]
fn distance_from_coords(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        Metric::Cosine => 1.0 - similarity::metrics::cosine_similarity(a, b),
        Metric::DotProduct => -similarity::metrics::dot_product(a, b),
        Metric::Euclidean => similarity::metrics::euclidean_distance(a, b),
    }
}

#[allow(clippy::too_many_arguments)]
fn search_layer_static(
    vectors: &[f32],
    nodes: &[Node],
    dimension: usize,
    metric: Metric,
    query: &[f32],
    entry_points: &[usize],
    ef: usize,
    layer: usize,
    scratch: &mut SearchScratch,
) -> Vec<Neighbor> {
    scratch.prepare(nodes.len());
    let visited = &mut scratch.visited;
    let visited_gen = scratch.visited_gen;
    let candidates = &mut scratch.candidates;
    let results = &mut scratch.results;

    for &ep in entry_points {
        if ep < visited.len() && visited[ep] != visited_gen {
            visited[ep] = visited_gen;
            let dist = distance_from_coords(metric, query, get_vector(vectors, dimension, ep));
            let n = Neighbor {
                id: ep,
                distance: dist,
            };
            candidates.push(Reverse(n));
            results.push(n);
            if results.len() > ef {
                results.pop();
            }
        }
    }

    while let Some(Reverse(closest)) = candidates.pop() {
        if let Some(furthest) = results.peek()
            && closest.distance > furthest.distance
            && results.len() >= ef
        {
            break;
        }

        if closest.id >= nodes.len() {
            continue;
        }

        let connections = &nodes[closest.id].connections;
        if layer >= connections.len() {
            continue;
        }

        for neighbor in &connections[layer] {
            let nid = neighbor.id;
            if nid < visited.len() && visited[nid] != visited_gen {
                visited[nid] = visited_gen;

                let dist = distance_from_coords(metric, query, get_vector(vectors, dimension, nid));

                if results.len() < ef || dist < results.peek().unwrap().distance {
                    let n = Neighbor {
                        id: nid,
                        distance: dist,
                    };
                    candidates.push(Reverse(n));
                    results.push(n);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
    }

    let mut result: Vec<Neighbor> = scratch.results.drain().collect();
    result.sort_unstable();
    result
}

fn select_neighbors_heuristic(
    vectors: &[f32],
    dimension: usize,
    metric: Metric,
    candidates: &mut [Neighbor],
    m: usize,
) -> Vec<Neighbor> {
    candidates.sort_unstable();

    if candidates.len() <= m {
        return candidates.to_vec();
    }

    let mut result: Vec<Neighbor> = Vec::with_capacity(m);

    for &c in candidates.iter() {
        if result.len() >= m {
            break;
        }

        let dominated = result.iter().any(|r| {
            let dist_cr = distance_from_coords(
                metric,
                get_vector(vectors, dimension, c.id),
                get_vector(vectors, dimension, r.id),
            );
            dist_cr < c.distance
        });

        if !dominated {
            result.push(c);
        }
    }
    result
}

pub struct HnswIndex {
    vectors: VectorStorage,
    nodes: Vec<Node>,
    entry_point: Option<usize>,
    max_layer: usize,
    dimension: usize,
    metric: Metric,
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    ef_search: usize,
    level_mult: f64,
    insert_scratch: SearchScratch,
}

impl HnswIndex {
    pub fn new(dimension: usize, metric: Metric, m: usize, ef_construction: usize) -> Self {
        assert!(m > 1, "M must be greater than 1");
        assert!(
            ef_construction > 0,
            "EF Construction must be greater than 0"
        );

        Self {
            vectors: VectorStorage::Owned(Vec::new()),
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
            dimension,
            metric,
            m,
            m_max0: m * 2,
            ef_construction,
            ef_search: ef_construction,
            level_mult: 1.0 / (m as f64).ln(),
            insert_scratch: SearchScratch::new(),
        }
    }

    pub fn set_ef_search(&mut self, ef: usize) {
        self.ef_search = ef;
    }

    fn distance_to_score(&self, distance: f32) -> f32 {
        match self.metric {
            Metric::Cosine => 1.0 - distance,
            Metric::DotProduct => -distance,
            Metric::Euclidean => 1.0 / (1.0 + distance),
        }
    }

    fn random_level(&self) -> usize {
        let r: f64 = rand::rng().random_range(0.0..1.0);
        (-r.ln() * self.level_mult).floor() as usize
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_snapshot(
        dimension: usize,
        metric: Metric,
        vectors: VectorStorage,
        nodes: Vec<NodeSnapshot>,
        entry_point: Option<usize>,
        max_layer: usize,
        m: usize,
        m_max0: usize,
        ef_construction: usize,
        ef_search: usize,
        level_mult: f64,
    ) -> Self {
        let nodes = nodes
            .into_iter()
            .map(|ns| Node {
                connections: ns
                    .connections
                    .into_iter()
                    .map(|layer| {
                        layer
                            .into_iter()
                            .map(|nb| Neighbor {
                                id: nb.id,
                                distance: nb.distance,
                            })
                            .collect()
                    })
                    .collect(),
            })
            .collect();

        Self {
            vectors,
            nodes,
            entry_point,
            max_layer,
            dimension,
            metric,
            m,
            m_max0,
            ef_construction,
            ef_search,
            level_mult,
            insert_scratch: SearchScratch::new(),
        }
    }
}

impl Index for HnswIndex {
    fn insert(&mut self, vector: Vec<f32>) {
        assert_eq!(
            vector.len(),
            self.dimension,
            "Vector dimension does not match index dimension"
        );

        self.vectors.ensure_owned();
        let id = self.vectors.as_slice().len() / self.dimension;
        self.vectors.as_owned_mut().extend_from_slice(&vector);

        let level = self.random_level();
        let mut new_connections = vec![Vec::new(); level + 1];

        if let Some(ep) = self.entry_point {
            let mut current_ep = vec![ep];

            for l in ((level + 1)..=self.max_layer).rev() {
                let result = search_layer_static(
                    self.vectors.as_slice(),
                    &self.nodes,
                    self.dimension,
                    self.metric,
                    &vector,
                    &current_ep,
                    1,
                    l,
                    &mut self.insert_scratch,
                );
                if !result.is_empty() {
                    current_ep = vec![result[0].id];
                }
            }

            let top = level.min(self.max_layer);
            for l in (0..=top).rev() {
                let mut neighbors = search_layer_static(
                    self.vectors.as_slice(),
                    &self.nodes,
                    self.dimension,
                    self.metric,
                    &vector,
                    &current_ep,
                    self.ef_construction,
                    l,
                    &mut self.insert_scratch,
                );

                let m = if l == 0 { self.m_max0 } else { self.m };

                let selected = select_neighbors_heuristic(
                    self.vectors.as_slice(),
                    self.dimension,
                    self.metric,
                    &mut neighbors,
                    m,
                );

                current_ep = selected.iter().map(|n| n.id).collect();
                new_connections[l] = selected;
            }

            self.nodes.push(Node {
                connections: new_connections,
            });

            for l in 0..=top {
                let neighbors_to_update: Vec<Neighbor> = self.nodes[id].connections[l].clone();

                for neighbor in neighbors_to_update {
                    let neighbor_id = neighbor.id;
                    let back_link = Neighbor {
                        id,
                        distance: neighbor.distance,
                    };

                    if l < self.nodes[neighbor_id].connections.len() {
                        self.nodes[neighbor_id].connections[l].push(back_link);

                        let max_conn = if l == 0 { self.m_max0 } else { self.m };
                        if self.nodes[neighbor_id].connections[l].len() > max_conn {
                            let mut candidates =
                                std::mem::take(&mut self.nodes[neighbor_id].connections[l]);
                            let pruned = select_neighbors_heuristic(
                                self.vectors.as_slice(),
                                self.dimension,
                                self.metric,
                                &mut candidates,
                                max_conn,
                            );
                            self.nodes[neighbor_id].connections[l] = pruned;
                        }
                    }
                }
            }

            if level > self.max_layer {
                self.max_layer = level;
                self.entry_point = Some(id);
            }
        } else {
            self.nodes.push(Node {
                connections: new_connections,
            });
            self.entry_point = Some(id);
            self.max_layer = level;
        }
    }

    fn search(&self, query: Vec<f32>, k: usize) -> Vec<IndexSearchResult> {
        assert_eq!(
            query.len(),
            self.dimension,
            "Query dimension does not match index dimension"
        );

        if self.entry_point.is_none() || k == 0 {
            return Vec::new();
        }

        let vectors = self.vectors.as_slice();
        let mut scratch = SearchScratch::new();
        let ep = self.entry_point.unwrap();
        let mut current_ep = vec![ep];

        for l in (1..=self.max_layer).rev() {
            let result = search_layer_static(
                vectors,
                &self.nodes,
                self.dimension,
                self.metric,
                &query,
                &current_ep,
                1,
                l,
                &mut scratch,
            );
            if !result.is_empty() {
                current_ep = vec![result[0].id];
            }
        }

        let ef = self.ef_search.max(k);
        let results = search_layer_static(
            vectors,
            &self.nodes,
            self.dimension,
            self.metric,
            &query,
            &current_ep,
            ef,
            0,
            &mut scratch,
        );

        results
            .iter()
            .take(k)
            .map(|n| IndexSearchResult {
                id: n.id,
                score: self.distance_to_score(n.distance),
            })
            .collect()
    }

    fn get(&self, id: usize) -> Option<&[f32]> {
        let vectors = self.vectors.as_slice();
        let start = id.checked_mul(self.dimension)?;
        let end = start.checked_add(self.dimension)?;
        vectors.get(start..end)
    }

    fn len(&self) -> usize {
        self.vectors.as_slice().len() / self.dimension
    }

    fn snapshot(&self) -> IndexSnapshot {
        IndexSnapshot::Hnsw {
            vectors: VectorStorage::Owned(self.vectors.as_slice().to_vec()),
            nodes: self
                .nodes
                .iter()
                .map(|n| NodeSnapshot {
                    connections: n
                        .connections
                        .iter()
                        .map(|layer| {
                            layer
                                .iter()
                                .map(|nb| NeighborSnapshot {
                                    id: nb.id,
                                    distance: nb.distance,
                                })
                                .collect()
                        })
                        .collect(),
                })
                .collect(),
            entry_point: self.entry_point,
            max_layer: self.max_layer,
            m: self.m,
            m_max0: self.m_max0,
            ef_construction: self.ef_construction,
            ef_search: self.ef_search,
            level_mult: self.level_mult,
        }
    }
}
