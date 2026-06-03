use std::fs;

use crate::{DIMENSIONS, LABELS_PATH, VECTORS_PATH};

pub struct Reference {
    pub vectors: Box<[[u16; DIMENSIONS]]>,
    pub labels: Box<[u64]>,
}

impl Reference {
    pub fn new() -> Self {
        let vector_bytes = fs::read(VECTORS_PATH).unwrap();

        let vectors = vector_bytes
            .chunks_exact(14 * 2)
            .map(|chunk| {
                let mut v = [0u16; 14];

                for i in 0..14 {
                    v[i] = u16::from_le_bytes(chunk[i * 2..i * 2 + 2].try_into().unwrap());
                }

                v
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let label_bytes = fs::read(LABELS_PATH).unwrap();

        let labels = label_bytes
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self { vectors, labels }
    }

    #[inline]
    pub fn squared_euclidean_distance(a: &[u16; DIMENSIONS], b: &[u16; DIMENSIONS]) -> u64 {
        let mut sum = 0u64;

        for i in 0..DIMENSIONS {
            let diff = a[i] as i32 - b[i] as i32;
            sum += (diff * diff) as u64;
        }

        sum
    }

    pub fn fraud_score(&self, transaction: &[u16; DIMENSIONS]) -> f64 {
        let mut best = [(u64::MAX, 0usize); 5];

        for (idx, vector) in self.vectors.iter().enumerate() {
            let sum = Self::squared_euclidean_distance(transaction, vector);

            // Possibly hot path to skip with few comparisons
            if sum >= best[4].0 {
                continue;
            }

            best[4] = (sum, idx);

            let mut i = 4;
            while i > 0 && best[i].0 < best[i - 1].0 {
                best.swap(i, i - 1);
                i -= 1;
            }
        }

        let mut fraud_count: f64 = 0.0;

        // Resole idx to labels
        for (_, idx) in best {
            let word_idx = idx >> 6;
            let bit_idx = idx & 63;

            if ((self.labels[word_idx] >> bit_idx) & 1) != 0 {
                fraud_count += 1.0;
            }
        }

        fraud_count / 5.0
    }
}
