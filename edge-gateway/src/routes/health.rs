use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};
use chrono::Utc;

use crate::catalog::SharedCatalog;

pub async fn health_handler(State(catalog): State<SharedCatalog>) -> Json<Value> {
    let cat = catalog.read().await;
    let count = cat.profiles.len();

    Json(json!({
        "status": "healthy",
        "service": "ReturnShield AI Rust Edge Gateway",
        "version": env!("CARGO_PKG_VERSION"),
        "registered_skus_count": count,
        "runtime": "Native Rust (Tokio/Axum)",
        "concurrency": "Multi-Threaded Work-Stealing",
        "timestamp": Utc::now().to_rfc3339()
    }))
}
