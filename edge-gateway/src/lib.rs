//! ReturnShield AI Edge Gateway Library.

pub mod catalog;
pub mod models;
pub mod routes;
pub mod verification;

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;

use routes::*;

/// Builds the Axum router with all edge routes and shared state.
pub fn create_router(catalog: catalog::SharedCatalog) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/skus", get(list_skus_handler).post(onboard_sku_handler))
        .route("/skus/:sku_id", get(get_sku_handler))
        .route("/returns/inspect", post(inspect_return_handler))
        .route("/audit/verify", post(verify_certificate_handler))
        .layer(CorsLayer::permissive())
        .with_state(catalog)
}
