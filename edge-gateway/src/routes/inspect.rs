use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use std::time::Instant;

use crate::catalog::SharedCatalog;
use crate::models::{
    IdentityMatchStatus, ReturnInspectionRequest, ReturnInspectionResult, SwapDetectionResult,
};
use crate::verification::{DispositionEngine, KittingAuditor, SerialVerifier};

pub async fn inspect_return_handler(
    State(catalog): State<SharedCatalog>,
    Json(req): Json<ReturnInspectionRequest>,
) -> Result<Json<ReturnInspectionResult>, (StatusCode, String)> {
    let t_start = Instant::now();

    // 1. Catalog SKU profile lookup
    let cat = catalog.read().await;
    let profile = cat.get(&req.sku_id).cloned().ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("SKU '{}' is not registered in catalog.", req.sku_id),
        )
    })?;
    drop(cat);

    // 2. Dual-Serial Number Verification
    // Compares box barcode against trusted order-side expected customer serial.
    // Box barcode is never compared to itself as simulated chassis.
    let serial_result = if req.box_barcode_raw.is_some() || req.expected_customer_serial.is_some() {
        Some(SerialVerifier::verify(
            req.box_barcode_raw.as_deref(),
            req.expected_customer_serial.as_deref(),
            profile.expected_serial_regex.as_deref(),
        ))
    } else {
        None
    };

    // 3. Packaging Tray BOM Kitting Audit
    let kitting_auditor = KittingAuditor::default();
    let kitting_result = if req.tray_crop_path.is_some() {
        Some(kitting_auditor.audit(
            &profile.bom,
            None,
            req.tray_crop_path.as_deref(),
        ))
    } else {
        None
    };

    // 4. Visual Identity Confirmation (Baseline Edge Engine)
    // Edge Gateway marks unverified unless evaluated by a validated vision model worker.
    let swap_result = if req.chassis_crop_path.is_some() {
        Some(SwapDetectionResult {
            identity_status: IdentityMatchStatus::Unverified,
            cosine_similarity: 0.0,
            threshold: profile.similarity_threshold,
            is_swap_detected: false,
            explanation: "Edge Gateway received chassis crop; unverified at edge until evaluated by primary vision worker.".to_string(),
            spatial_diff_score: None,
            heatmap_path: None,
        })
    } else {
        None
    };

    // 5. Algorithmic Financial Disposition & SHA-256 Certificate Sealing
    let surface_wear_score = 0.0;
    let (disposition, certificate) = DispositionEngine::calculate_disposition(
        &profile,
        &req.rma_number,
        swap_result.as_ref(),
        serial_result.as_ref(),
        kitting_result.as_ref(),
        surface_wear_score,
        None,
    );

    let elapsed = t_start.elapsed().as_secs_f64() * 1000.0;
    let fraud_flags = certificate.fraud_flags.clone();
    let is_fraud = disposition.action == crate::models::DispositionAction::RejectFraudHold;

    let result = ReturnInspectionResult {
        inspection_id: format!("INSP-{}", uuid::Uuid::new_v4().simple().to_string()[..10].to_uppercase()),
        rma_number: req.rma_number,
        sku_id: req.sku_id,
        timestamp: chrono::Utc::now(),
        fraud_flags,
        is_fraud,
        swap_result,
        serial_result,
        kitting_result,
        disposition,
        certificate,
        processing_time_ms: (elapsed * 100.0).round() / 100.0,
        operator_id: req.operator_id,
        bench_id: req.bench_id,
    };

    Ok(Json(result))
}

