use std::arch::x86_64::*;
use std::fs;

use crate::{DIMENSIONS, LABELS_PATH, VECTORS_PATH};

const LANES: usize = 16;

type Block = [[u16; LANES]; DIMENSIONS];

pub struct ReferenceSIMD {
    pub blocks: Box<[Block]>,
    pub labels: Box<[u64]>,
    pub len: usize, // real number of vectors, excluding padding
}

impl ReferenceSIMD {
    pub fn new(len: usize) -> Self {
        let vector_bytes = fs::read(VECTORS_PATH).unwrap();

        let block_size_bytes = DIMENSIONS * LANES * 2;

        let blocks = vector_bytes
            .chunks_exact(block_size_bytes)
            .map(|chunk| {
                let mut block = [[0u16; LANES]; DIMENSIONS];

                for dim in 0..DIMENSIONS {
                    for lane in 0..LANES {
                        let offset = (dim * LANES + lane) * 2;

                        block[dim][lane] = u16::from_le_bytes(
                            chunk[offset..offset + 2].try_into().unwrap(),
                        );
                    }
                }

                block
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let label_bytes = fs::read(LABELS_PATH).unwrap();

        let labels = label_bytes
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            blocks,
            labels,
            len,
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn distances_16(transaction: &[u16; DIMENSIONS], block: &Block) -> [u32; LANES] {
        let mut acc_lo = _mm256_setzero_si256(); // lanes 0..7
        let mut acc_hi = _mm256_setzero_si256(); // lanes 8..15

        for dim in 0..DIMENSIONS {
            let values = _mm256_loadu_si256(block[dim].as_ptr() as *const __m256i);

            let lo_128 = _mm256_castsi256_si128(values);
            let hi_128 = _mm256_extracti128_si256(values, 1);

            let vals_lo = _mm256_cvtepu16_epi32(lo_128);
            let vals_hi = _mm256_cvtepu16_epi32(hi_128);

            let q = _mm256_set1_epi32(transaction[dim] as i32);

            let diff_lo = _mm256_sub_epi32(q, vals_lo);
            let diff_hi = _mm256_sub_epi32(q, vals_hi);

            let sq_lo = _mm256_mullo_epi32(diff_lo, diff_lo);
            let sq_hi = _mm256_mullo_epi32(diff_hi, diff_hi);

            acc_lo = _mm256_add_epi32(acc_lo, sq_lo);
            acc_hi = _mm256_add_epi32(acc_hi, sq_hi);
        }

        let mut out = [0u32; LANES];

        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, acc_lo);
        _mm256_storeu_si256(out.as_mut_ptr().add(8) as *mut __m256i, acc_hi);

        out
    }

    #[inline]
    fn insert_best(best: &mut [(u64, usize); 5], distance: u64, idx: usize) {
        if distance >= best[4].0 {
            return;
        }

        best[4] = (distance, idx);

        let mut i = 4;
        while i > 0 && best[i].0 < best[i - 1].0 {
            best.swap(i, i - 1);
            i -= 1;
        }
    }

    pub fn fraud_score(&self, transaction: &[u16; DIMENSIONS]) -> f64 {
        let mut best = [(u64::MAX, 0usize); 5];

        for (block_idx, block) in self.blocks.iter().enumerate() {
            let block_start = block_idx * LANES;

            let real_count = if block_start + LANES <= self.len {
                LANES
            } else {
                self.len - block_start
            };

            let distances = unsafe { Self::distances_16(transaction, block) };

            for lane in 0..real_count {
                let idx = block_start + lane;
                let distance = distances[lane] as u64;

                Self::insert_best(&mut best, distance, idx);
            }
        }

        let mut fraud_count = 0usize;

        for (_, idx) in best {
            let word_idx = idx >> 6;
            let bit_idx = idx & 63;

            if ((self.labels[word_idx] >> bit_idx) & 1) != 0 {
                fraud_count += 1;
            }
        }

        fraud_count as f64 / 5.0
    }
}