//! ReturnShield AI Edge Gateway Library.

pub mod catalog;
pub mod models;
pub mod routes;
pub mod verification;

use axum::http::{HeaderValue, Method};
use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::{AllowOrigin, CorsLayer};

use routes::*;

/// Builds configured CORS layer restricted to dashboard and known origins.
pub fn build_cors_layer() -> CorsLayer {
    let origins_env = std::env::var("RETURNSHIELD_ALLOWED_ORIGINS").unwrap_or_else(|_| {
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000".to_string()
    });

    let origins: Vec<HeaderValue> = origins_env
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<HeaderValue>().ok())
        .collect();

    CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([
            axum::http::header::CONTENT_TYPE,
            axum::http::header::AUTHORIZATION,
            axum::http::header::HeaderName::from_static("x-api-key"),
            axum::http::header::HeaderName::from_static("x-tenant-id"),
        ])
}

/// Builds the Axum router with all edge routes and shared state.
pub fn create_router(catalog: catalog::SharedCatalog) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/skus", get(list_skus_handler).post(onboard_sku_handler))
        .route("/skus/:sku_id", get(get_sku_handler))
        .route("/returns/inspect", post(inspect_return_handler))
        .route("/audit/verify", post(verify_certificate_handler))
        .layer(build_cors_layer())
        .with_state(catalog)
}
