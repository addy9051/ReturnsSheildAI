//! High-Throughput Concurrent SKU Catalog and Embedding Cache in Rust.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::models::SkuProfile;

pub struct SkuCatalog {
    pub profiles: HashMap<String, SkuProfile>,
    pub db_path: PathBuf,
}

impl SkuCatalog {
    /// Loads registered catalog profiles from disk database.
    pub fn load_from_disk(db_path: &Path) -> Self {
        let mut profiles = HashMap::new();

        if db_path.exists() {
            if let Ok(file) = File::open(db_path) {
                let reader = BufReader::new(file);
                match serde_json::from_reader::<_, HashMap<String, SkuProfile>>(reader) {
                    Ok(map) => {
                        profiles = map;
                        info!("Loaded {} registered SKUs into Edge Catalog memory.", profiles.len());
                    }
                    Err(err) => {
                        warn!("Failed parsing catalog_db.json into SkuProfile schema: {}", err);
                    }
                }
            }
        } else {
            warn!("Catalog DB path not found: {:?}. Starting with empty catalog.", db_path);
        }

        Self {
            profiles,
            db_path: db_path.to_path_buf(),
        }
    }

    pub fn get(&self, sku_id: &str) -> Option<&SkuProfile> {
        self.profiles.get(sku_id)
    }

    pub fn list(&self) -> Vec<SkuProfile> {
        self.profiles.values().cloned().collect()
    }

    pub fn insert(&mut self, profile: SkuProfile) {
        self.profiles.insert(profile.sku_id.clone(), profile);
    }

    /// Computes cosine similarity between two L2-normalized visual embedding vectors.
    pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f64 {
        if v1.len() != v2.len() || v1.is_empty() {
            return 0.0;
        }

        let mut dot = 0.0f64;
        let mut norm1 = 0.0f64;
        let mut norm2 = 0.0f64;

        for (a, b) in v1.iter().zip(v2.iter()) {
            let fa = *a as f64;
            let fb = *b as f64;
            dot += fa * fb;
            norm1 += fa * fa;
            norm2 += fb * fb;
        }

        let denom = (norm1.sqrt()) * (norm2.sqrt());
        if denom > 1e-8 {
            dot / denom
        } else {
            0.0
        }
    }
}

pub type SharedCatalog = Arc<RwLock<SkuCatalog>>;
