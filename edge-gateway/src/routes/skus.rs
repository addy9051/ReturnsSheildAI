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
) -> (StatusCode, Json<SkuProfile>) {
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

    cat.insert(profile.clone());
    (StatusCode::CREATED, Json(profile))
}
