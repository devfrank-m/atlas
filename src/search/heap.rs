use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Debug)]
pub struct ScoredItem<T> {
    pub score: f32,
    pub item: T,
}

impl<T> PartialEq for ScoredItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl<T> Eq for ScoredItem<T> {}

impl<T> Ord for ScoredItem<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl<T> PartialOrd for ScoredItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub struct TopKHeap<T> {
    heap: BinaryHeap<Reverse<ScoredItem<T>>>,
    k: usize,
}

impl<T> TopKHeap<T> {
    pub fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::new(),
            k,
        }
    }

    pub fn push(&mut self, score: f32, item: T) {
        if self.heap.len() < self.k {
            // we continue to push until we have k items
            self.heap.push(Reverse(ScoredItem { score, item }));
        } else {
            // if the new score is better than the smallest in our Top-K, then we pop and push
            match self.heap.peek() {
                Some(Reverse(min)) if score > min.score => {
                    self.heap.pop();
                    self.heap.push(Reverse(ScoredItem { score, item }));
                }
                _ => {}
            }
        }
    }

    pub fn into_sorted_vec(self) -> Vec<T> {
        let sorted_heap = self.heap.into_sorted_vec();
        let mut sorted_vec = Vec::with_capacity(sorted_heap.len());

        for Reverse(scored_item) in sorted_heap {
            sorted_vec.push(scored_item.item);
        }

        sorted_vec
    }
}
