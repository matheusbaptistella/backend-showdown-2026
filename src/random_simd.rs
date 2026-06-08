use std::arch::x86_64::*;
use std::fs;

use crate::{DIMENSIONS, LABELS_PATH, VECTORS_PATH};

const TOTAL_AMOUNT_VECTORS: u64 = 3_000_000;
const VECTORS_PER_BLOCK: usize = 16;
const VECTOR_SAMPLE: u64 = 32_768;

const BLOCK_SAMPLE: u64 = VECTOR_SAMPLE / VECTORS_PER_BLOCK as u64;
const TOTAL_AMOUNT_BLOCKS: u64 = TOTAL_AMOUNT_VECTORS / VECTORS_PER_BLOCK as u64;
const BLOCK_SIZE_BYTES: usize = VECTORS_PER_BLOCK * DIMENSIONS * 2;
const BLOCK_CHOICES_O: usize = (TOTAL_AMOUNT_BLOCKS - BLOCK_SAMPLE + 1) as usize;

type Block = [[u16; VECTORS_PER_BLOCK]; DIMENSIONS];

pub struct RandomSIMD {
    pub blocks: Box<[Block]>,
    pub labels: Box<[u64]>,
}

impl RandomSIMD {
    pub fn new() -> Self {
        let vector_bytes = fs::read(VECTORS_PATH).unwrap();

        let blocks = vector_bytes
            .chunks_exact(BLOCK_SIZE_BYTES)
            .map(|chunk| {
                let mut block = [[0u16; VECTORS_PER_BLOCK]; DIMENSIONS];

                for dim in 0..DIMENSIONS {
                    for lane in 0..VECTORS_PER_BLOCK {
                        let offset = (dim * VECTORS_PER_BLOCK + lane) * 2;

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
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn distances_16(transaction: &[u16; DIMENSIONS], block: &Block) -> [u32; VECTORS_PER_BLOCK] {
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

        let mut out = [0u32; VECTORS_PER_BLOCK];

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

    #[inline]
    fn fast_bound(rand: u64, n: usize) -> usize {
        (((rand as u128) * (n as u128)) >> 64) as usize
    }

    #[inline]
    fn sample_start_block(transaction: &[u16; DIMENSIONS]) -> usize {
        let mut h = 0x9E37_79B9_7F4A_7C15u64;

        for (i, &v) in transaction.iter().enumerate() {
            h ^= (v as u64).wrapping_add((i as u64) << 32);
            h = h.rotate_left(27);
            h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
        }

        h = h ^ (h >> 33);

        Self::fast_bound(h, BLOCK_CHOICES_O)
    }

    pub fn fraud_score(&self, transaction: &[u16; DIMENSIONS]) -> f64 {
        let mut best = [(u64::MAX, 0usize); 5];

        let start_block = Self::sample_start_block(transaction);
        let end_block = start_block + BLOCK_SAMPLE as usize;

        let blocks = &self.blocks[start_block..end_block];

        for (offset, block) in blocks.iter().enumerate() {
            let distances = unsafe { Self::distances_16(transaction, block) };

            let block_idx = start_block + offset;
            let block_start = block_idx * VECTORS_PER_BLOCK;

            for i in 0..VECTORS_PER_BLOCK {
                let idx = block_start + i;
                let distance = distances[i] as u64;

                Self::insert_best(&mut best, distance, idx);
            }
        }

        let mut fraud_count = 0f64;

        for (_, idx) in best {
            let word_idx = idx >> 6;
            let bit_idx = idx & 63;

            fraud_count += ((self.labels[word_idx] >> bit_idx) & 1) as f64;
        }

        fraud_count / 5.0
    }

}