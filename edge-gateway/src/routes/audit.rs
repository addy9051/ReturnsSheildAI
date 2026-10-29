use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Debug, Deserialize)]
pub struct VerifyRequest {
    pub dispute_ready_payload: Value,
    pub evidence_hash: String,
}

#[derive(Debug, Serialize)]
pub struct VerifyResponse {
    pub is_valid: bool,
    pub provided_hash: String,
    pub computed_hash: String,
    pub tamper_detected: bool,
    pub status: String,
}

pub async fn verify_certificate_handler(Json(req): Json<VerifyRequest>) -> Json<VerifyResponse> {
    let mut hasher = Sha256::new();
    hasher.update(req.dispute_ready_payload.to_string().as_bytes());
    let computed_hash = format!("{:x}", hasher.finalize());

    let is_valid = computed_hash == req.evidence_hash;

    Json(VerifyResponse {
        is_valid,
        provided_hash: req.evidence_hash,
        computed_hash,
        tamper_detected: !is_valid,
        status: if is_valid {
            "AUTHENTIC_VERIFIED".to_string()
        } else {
            "TAMPER_DETECTED_INVALID".to_string()
        },
    })
}

