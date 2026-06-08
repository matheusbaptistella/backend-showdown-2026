use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    response::IntoResponse,
    routing::{get, post},
};
use backend_showdown_2026::{FraudScorePayload, FraudScoreResponse, Reference, ReferenceSIMD, Vector};

#[tokio::main(flavor = "multi_thread", worker_threads = 1)]
// Compare vs #[tokio::main(flavor = "current_thread")]
async fn main() {
    // let state = Arc::new(Reference::new()); // Must be cheap to clone
    let state = Arc::new(ReferenceSIMD::new(3_000_000));

    let app = Router::new()
        .route("/ready", get(ready))
        .route("/fraud-score", post(fraud_score))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();

    println!("Listening on 0.0.0.0:3000");

    axum::serve(listener, app).await.unwrap();
}

async fn ready() {}

async fn fraud_score(
    // State(reference): State<Arc<Reference>>,
    State(reference): State<Arc<ReferenceSIMD>>,
    Json(payload): Json<FraudScorePayload>,
) -> impl IntoResponse {
    let vector: Vector = payload.into();

    let fraud_score = reference.fraud_score(&vector);

    let approved = if fraud_score < 0.6 { true } else { false };

    Json(FraudScoreResponse {
        approved,
        fraud_score,
    })
}
