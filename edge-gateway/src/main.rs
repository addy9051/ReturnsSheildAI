use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use returnshield_edge::catalog::SkuCatalog;
use returnshield_edge::create_router;

#[derive(Parser, Debug)]
#[command(name = "returnshield-edge", author, version, about = "ReturnShield AI High-Speed Rust Edge Gateway", long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    #[arg(short, long, default_value = "../storage/catalog_db.json")]
    catalog_db: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "returnshield_edge=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();
    info!("Starting ReturnShield AI Edge Gateway on port {}...", args.port);

    // Initialize in-memory catalog
    let catalog = SkuCatalog::load_from_disk(&args.catalog_db);
    let shared_catalog = Arc::new(RwLock::new(catalog));

    let app = create_router(shared_catalog);
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;

    info!("ReturnShield Edge Gateway listening on http://{}", addr);
    axum::serve(listener, app).await?;

    Ok(())
}
