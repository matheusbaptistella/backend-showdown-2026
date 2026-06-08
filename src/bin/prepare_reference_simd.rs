use serde::Deserialize;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

use backend_showdown_2026::{DIMENSIONS, OFFSET, SCALE, LANES, INPUT_JSON};

const VECTORS_PATH: &str = "references.vectors.u16.simd";
const LABELS_PATH: &str = "references.labels.bits.simd";

#[derive(Debug, Deserialize)]
struct Record {
    vector: [f64; DIMENSIONS],
    label: String,
}

fn encode_value(value: f64) -> u16 {
    ((value + OFFSET) * SCALE).round() as u16
}

fn main() {
    let file = File::open(INPUT_JSON).unwrap();
    let reader = BufReader::new(file);

    let records: Vec<Record> = serde_json::from_reader(reader).unwrap();

    let mut vectors_writer = BufWriter::new(File::create(VECTORS_PATH).unwrap());

    let label_words_len = (records.len() + 63) / 64;
    let mut label_words = vec![0u64; label_words_len];

    // Labels stay indexed by original record index.
    for (idx, record) in records.iter().enumerate() {
        if record.label == "fraud" {
            let word_idx = idx >> 6;
            let bit_idx = idx & 63;
            label_words[word_idx] |= 1u64 << bit_idx;
        }
    }

    // Write vectors as blocks of 16 records:
    //
    // block 0:
    //   dim0: v0.d0,  v1.d0,  ... v15.d0
    //   dim1: v0.d1,  v1.d1,  ... v15.d1
    //   ...
    //   dim13: v0.d13, ... v15.d13
    //
    // block 1:
    //   dim0: v16.d0, v17.d0, ... v31.d0
    //   ...
    for block_start in (0..records.len()).step_by(LANES) {
        for dim in 0..DIMENSIONS {
            for lane in 0..LANES {
                let idx = block_start + lane;

                let encoded = if idx < records.len() {
                    encode_value(records[idx].vector[dim])
                } else {
                    0
                };

                vectors_writer.write_all(&encoded.to_le_bytes()).unwrap();
            }
        }
    }

    vectors_writer.flush().unwrap();

    let mut labels_writer = BufWriter::new(File::create(LABELS_PATH).unwrap());

    for word in label_words {
        labels_writer.write_all(&word.to_le_bytes()).unwrap();
    }

    labels_writer.flush().unwrap();

    println!("converted {} records", records.len());
    println!("vectors written to {}", VECTORS_PATH);
    println!("labels written to {}", LABELS_PATH);
}