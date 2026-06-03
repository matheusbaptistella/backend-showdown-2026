use serde::Deserialize;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

use backend_showdown_2026::{DIMENSIONS, LABELS_PATH, OFFSET, SCALE, VECTORS_PATH};

const INPUT_JSON: &str =
    "/Users/baptistellamatheus/projects/rinha-de-backend-2026/resources/references.json";

#[derive(Debug, Deserialize)]
struct Record {
    vector: [f64; DIMENSIONS],
    label: String,
}

fn encode_value(value: f64) -> u16 {
    ((value + OFFSET) * SCALE).round() as u16 // Maybe dont need round()
}

fn main() {
    let file = File::open(INPUT_JSON).unwrap();
    let reader = BufReader::new(file);

    let records: Vec<Record> = serde_json::from_reader(reader).unwrap();

    let mut vectors_writer = BufWriter::new(File::create(VECTORS_PATH).unwrap());

    let label_words_len = (records.len() + 63) / 64;
    let mut label_words = vec![0u64; label_words_len];

    for (idx, record) in records.iter().enumerate() {
        for value in record.vector {
            let encoded = encode_value(value);
            vectors_writer.write_all(&encoded.to_le_bytes()).unwrap();
        }

        if record.label == "fraud" {
            let word_idx = idx >> 6;
            let bit_idx = idx & 63;
            label_words[word_idx] |= 1u64 << bit_idx;
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
