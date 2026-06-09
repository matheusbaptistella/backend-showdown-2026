use chrono::{DateTime, Datelike, Timelike, Utc};
use serde::{Deserialize, Serialize};

pub mod reference;
pub use reference::Reference;
pub mod reference_simd;
pub use reference_simd::ReferenceSIMD;
pub mod random_simd;
pub use random_simd::RandomSIMD;

const MAX_AMOUNT: f64 = 10_000.0;
const MAX_INSTALLMENTS: f64 = 12.0;
const AMOUNT_VS_AVG_RATIO: f64 = 10.0;
const MAX_MINUTES: f64 = 1440.0;
const MAX_KM: f64 = 1_000.0;
const MAX_TX_COUNT_24H: f64 = 20.0;
const MAX_MERCHANT_AVG_AMOUNT: f64 = 10_000.0;

const MISSING_LAST_TX: f64 = -1.0;

pub const DIMENSIONS: usize = 14;
pub const LANES: usize = 16;
pub const SCALE: f64 = 10_000.0;
pub const OFFSET: f64 = 1.0;

pub const VECTORS_PATH: &str = "references.vectors.u16.simd";
pub const LABELS_PATH: &str = "references.labels.bits.simd";
pub const INPUT_JSON: &str = "/home/matheus/projects/rinha-de-backend-2026/resources/references.json";

pub type Vector = [u16; DIMENSIONS];

#[inline]
fn mcc_risk(mcc: &str) -> f64 {
    match mcc {
        "4511" => 0.35,
        "5311" => 0.25,
        "5411" => 0.15,
        "5812" => 0.30,
        "5912" => 0.20,
        "5944" => 0.45,
        "5999" => 0.50,
        "7801" => 0.80,
        "7802" => 0.75,
        "7995" => 0.85,
        _ => 0.5,
    }
}

// Maybe serialize json to array (do not use struct simply the positions) to avoid allocations
#[derive(Deserialize)]
pub struct FraudScorePayload {
    #[serde(rename = "id")]
    _id: String,
    transaction: Transaction,
    customer: Customer,
    merchant: Merchant,
    terminal: Terminal,
    last_transaction: Option<LastTransaction>,
}

#[derive(Deserialize)]
pub struct Transaction {
    amount: f64,
    installments: u64,
    requested_at: DateTime<Utc>,
}

#[derive(Deserialize)]
pub struct Customer {
    avg_amount: f64,
    tx_count_24h: u64,
    known_merchants: Vec<String>,
}

#[derive(Deserialize)]
pub struct Merchant {
    id: String,
    mcc: String,
    avg_amount: f64,
}

#[derive(Deserialize)]
pub struct Terminal {
    is_online: bool,
    card_present: bool,
    km_from_home: f64,
}

#[derive(Deserialize)]
pub struct LastTransaction {
    timestamp: DateTime<Utc>,
    km_from_current: f64,
}

#[derive(Serialize)]
pub struct FraudScoreResponse {
    pub approved: bool,
    pub fraud_score: f64,
}

impl From<FraudScorePayload> for Vector {
    fn from(value: FraudScorePayload) -> Self {
        [
            (value.transaction.amount / MAX_AMOUNT).clamp(0.0, 1.0), // Maybe check if should handle negative cases
            (value.transaction.installments as f64 / MAX_INSTALLMENTS).clamp(0.0, 1.0),
            ((value.transaction.amount / value.customer.avg_amount) / AMOUNT_VS_AVG_RATIO)
                .clamp(0.0, 1.0),
            (value.transaction.requested_at.hour() as f64 / 23.0),
            (value
                .transaction
                .requested_at
                .weekday()
                .num_days_from_monday() as f64
                / 6.0),
            value
                .last_transaction
                .as_ref()
                .map_or(MISSING_LAST_TX, |tx| {
                    ((value.transaction.requested_at - tx.timestamp).num_minutes() as f64 / MAX_MINUTES).clamp(0.0, 1.0)
                }),
            value
                .last_transaction
                .as_ref()
                .map_or(MISSING_LAST_TX, |tx| {
                    (tx.km_from_current / MAX_KM).clamp(0.0, 1.0)
                }),
            (value.terminal.km_from_home / MAX_KM).clamp(0.0, 1.0),
            (value.customer.tx_count_24h as f64 / MAX_TX_COUNT_24H).clamp(0.0, 1.0),
            if value.terminal.is_online { 1.0 } else { 0.0 },
            if value.terminal.card_present {
                1.0
            } else {
                0.0
            },
            if value.customer.known_merchants.contains(&value.merchant.id) {
                0.0
            } else {
                1.0
            },
            mcc_risk(&value.merchant.mcc),
            (value.merchant.avg_amount / MAX_MERCHANT_AVG_AMOUNT).clamp(0.0, 1.0),
        ]
        .map(|value| ((value + OFFSET) * SCALE) as u16)
    }
}
