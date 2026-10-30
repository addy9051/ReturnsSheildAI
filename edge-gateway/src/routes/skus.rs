use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::Json;

use crate::catalog::SharedCatalog;
use crate::models::{SkuOnboardRequest, SkuProfile};

pub async fn list_skus_handler(State(catalog): State<SharedCatalog>) -> Json<Vec<SkuProfile>> {
    let cat = catalog.read().await;
    Json(cat.list())
}

pub async fn get_sku_handler(
    State(catalog): State<SharedCatalog>,
    AxumPath(sku_id): AxumPath<String>,
) -> Result<Json<SkuProfile>, StatusCode> {
    let cat = catalog.read().await;
    match cat.get(&sku_id) {
        Some(profile) => Ok(Json(profile.clone())),
        None => Err(StatusCode::NOT_FOUND),
    }
}

pub async fn onboard_sku_handler(
    State(catalog): State<SharedCatalog>,
    Json(req): Json<SkuOnboardRequest>,
) -> Result<(StatusCode, Json<SkuProfile>), (StatusCode, String)> {
    // Validate SKU ID against path traversal and invalid characters
    let sku_regex = regex::Regex::new(r"^[A-Za-z0-9_-]{2,64}$").unwrap();
    if !sku_regex.is_match(&req.sku_id) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("Invalid SKU ID '{}'. Must match ^[A-Za-z0-9_-]{{2,64}}$", req.sku_id),
        ));
    }

    let mut cat = catalog.write().await;
    let profile = SkuProfile {
        sku_id: req.sku_id.clone(),
        title: req.title,
        category: req.category,
        msrp: req.msrp,
        restocking_fee_rate: req.restocking_fee_rate,
        expected_serial_regex: req.expected_serial_regex,
        reference_image_paths: req.reference_image_paths,
        bom: req.bom,
        similarity_threshold: req.similarity_threshold,
        created_at: Some(chrono::Utc::now().to_rfc3339()),
        updated_at: Some(chrono::Utc::now().to_rfc3339()),
        metadata: std::collections::HashMap::new(),
    };

    if let Err(e) = cat.insert(profile.clone()) {
        tracing::error!("Failed to persist SKU to disk: {}", e);
    }
    Ok((StatusCode::CREATED, Json(profile)))
}
