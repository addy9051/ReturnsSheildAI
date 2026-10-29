//! Core Data Contracts and Serialization Schemas for ReturnShield AI Edge Gateway.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::enums::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BOMItem {
    pub item_id: String,
    pub name: String,
    #[serde(default = "default_category")]
    pub category: String,
    #[serde(default = "default_qty")]
    pub required_qty: u32,
    #[serde(default = "default_fee")]
    pub deduction_fee: f64,
    pub expected_tray_slot: Option<String>,
    #[serde(default)]
    pub is_critical: bool,
}

fn default_category() -> String {
    "accessory".to_string()
}
fn default_qty() -> u32 {
    1
}
fn default_fee() -> f64 {
    15.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkuProfile {
    pub sku_id: String,
    pub title: String,
    pub category: String,
    pub msrp: f64,
    #[serde(default)]
    pub restocking_fee_rate: f64,
    #[serde(default = "default_serial_regex")]
    pub expected_serial_regex: Option<String>,
    #[serde(default)]
    pub reference_image_paths: Vec<String>,
    #[serde(default)]
    pub bom: Vec<BOMItem>,
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f64,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

fn default_serial_regex() -> Option<String> {
    Some(r"^[A-Z0-9]{8,18}$".to_string())
}
fn default_similarity_threshold() -> f64 {
    0.85
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkuOnboardRequest {
    pub sku_id: String,
    pub title: String,
    pub category: String,
    pub msrp: f64,
    #[serde(default)]
    pub restocking_fee_rate: f64,
    #[serde(default)]
    pub expected_serial_regex: Option<String>,
    #[serde(default)]
    pub bom: Vec<BOMItem>,
    #[serde(default)]
    pub reference_image_paths: Vec<String>,
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnInspectionRequest {
    pub rma_number: String,
    pub sku_id: String,
    pub expected_customer_serial: Option<String>,
    pub box_barcode_raw: Option<String>,
    pub overhead_image_path: Option<String>,
    pub chassis_crop_path: Option<String>,
    pub serial_plate_crop_path: Option<String>,
    pub tray_crop_path: Option<String>,
    #[serde(default = "default_operator")]
    pub operator_id: Option<String>,
    #[serde(default = "default_bench")]
    pub bench_id: Option<String>,
}

fn default_operator() -> Option<String> {
    Some("OPERATOR-EDGE-01".to_string())
}
fn default_bench() -> Option<String> {
    Some("BENCH-EDGE-01".to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemizedDeduction {
    pub reason: String,
    pub category: String,
    pub amount: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KittedItemAudit {
    pub bom_item_id: String,
    pub name: String,
    pub status: AccessoryStatus,
    pub confidence: f64,
    pub deduction_fee: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KittingAuditResult {
    pub expected_items_count: usize,
    pub detected_items_count: usize,
    pub missing_items_count: usize,
    pub all_items_present: bool,
    pub item_audits: Vec<KittedItemAudit>,
    pub total_accessory_deduction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerialVerificationResult {
    pub box_barcode: Option<String>,
    pub chassis_serial_raw: Option<String>,
    pub chassis_serial_normalized: Option<String>,
    pub match_status: SerialMatchStatus,
    pub is_match: bool,
    pub similarity_ratio: f64,
    pub ocr_confidence: f64,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapDetectionResult {
    pub identity_status: IdentityMatchStatus,
    pub cosine_similarity: f64,
    pub threshold: f64,
    pub is_swap_detected: bool,
    #[serde(default)]
    pub explanation: String,
    pub spatial_diff_score: Option<f64>,
    pub heatmap_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispositionResult {
    pub action: DispositionAction,
    pub condition_grade: ConditionGrade,
    pub original_msrp: f64,
    pub total_deductions: f64,
    pub net_refund_amount: f64,
    pub deduction_breakdown: Vec<ItemizedDeduction>,
    pub recommendation_summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvidenceCertificate {
    #[serde(default = "default_cert_id")]
    pub certificate_id: String,
    pub rma_number: String,
    pub sku_id: String,
    #[serde(default = "Utc::now")]
    pub timestamp: DateTime<Utc>,
    pub disposition_action: DispositionAction,
    pub condition_grade: ConditionGrade,
    pub fraud_flags: Vec<FraudFlag>,
    pub evidence_hash: String,
    #[serde(default)]
    pub evidence_snapshot_paths: HashMap<String, String>,
    pub audit_narrative: String,
    #[serde(default)]
    pub dispute_ready_payload: serde_json::Value,
}

fn default_cert_id() -> String {
    format!("CERT-{}", Uuid::new_v4().simple().to_string()[..12].to_uppercase())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnInspectionResult {
    #[serde(default = "default_insp_id")]
    pub inspection_id: String,
    pub rma_number: String,
    pub sku_id: String,
    #[serde(default = "Utc::now")]
    pub timestamp: DateTime<Utc>,
    pub fraud_flags: Vec<FraudFlag>,
    pub is_fraud: bool,
    pub swap_result: Option<SwapDetectionResult>,
    pub serial_result: Option<SerialVerificationResult>,
    pub kitting_result: Option<KittingAuditResult>,
    pub disposition: DispositionResult,
    pub certificate: AuditEvidenceCertificate,
    pub processing_time_ms: f64,
    pub operator_id: Option<String>,
    pub bench_id: Option<String>,
}

fn default_insp_id() -> String {
    format!("INSP-{}", Uuid::new_v4().simple().to_string()[..10].to_uppercase())
}
