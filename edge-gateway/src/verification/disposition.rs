//! Financial Disposition and Cryptographic Dispute Evidence Engine in Rust.

use chrono::Utc;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

use crate::models::{
    AccessoryStatus, AuditEvidenceCertificate, ConditionGrade, DispositionAction, DispositionResult,
    FraudFlag, IdentityMatchStatus, ItemizedDeduction, KittingAuditResult, SerialMatchStatus,
    SerialVerificationResult, SkuProfile, SwapDetectionResult,
};

pub struct DispositionEngine;

impl DispositionEngine {
    pub fn determine_condition_grade(surface_wear_score: f64, is_fraud: bool) -> ConditionGrade {
        if is_fraud {
            return ConditionGrade::FraudHold;
        }
        if surface_wear_score <= 0.10 {
            ConditionGrade::GradeA
        } else if surface_wear_score <= 0.30 {
            ConditionGrade::GradeB
        } else {
            ConditionGrade::GradeC
        }
    }

    pub fn evaluate_fraud_flags(
        swap_result: Option<&SwapDetectionResult>,
        serial_result: Option<&SerialVerificationResult>,
        kitting_result: Option<&KittingAuditResult>,
        condition_grade: ConditionGrade,
    ) -> Vec<FraudFlag> {
        let mut flags = Vec::new();
        let mut has_incomplete_evidence = false;

        if let Some(res) = swap_result {
            if res.identity_status == IdentityMatchStatus::ConfirmedSwap {
                flags.push(FraudFlag::BaitAndSwitch);
            } else if res.identity_status == IdentityMatchStatus::Unverified {
                has_incomplete_evidence = true;
            }
        } else {
            has_incomplete_evidence = true;
        }

        if let Some(res) = serial_result {
            if !res.is_match {
                flags.push(FraudFlag::SerialMismatch);
            }
            if res.match_status == SerialMatchStatus::OcrUnreadable {
                flags.push(FraudFlag::TamperedSerialLabel);
            }
            if res.match_status == SerialMatchStatus::MissingLabel {
                has_incomplete_evidence = true;
            }
        } else {
            has_incomplete_evidence = true;
        }

        if let Some(res) = kitting_result {
            if !res.all_items_present {
                flags.push(FraudFlag::MissingAccessory);
            }
        }

        if condition_grade == ConditionGrade::GradeC {
            flags.push(FraudFlag::PhysicalDamage);
        }

        if has_incomplete_evidence {
            flags.push(FraudFlag::IncompleteEvidence);
        }

        if flags.is_empty() {
            flags.push(FraudFlag::Clean);
        }

        flags
    }

    pub fn calculate_disposition(
        sku_profile: &SkuProfile,
        rma_number: &str,
        swap_result: Option<&SwapDetectionResult>,
        serial_result: Option<&SerialVerificationResult>,
        kitting_result: Option<&KittingAuditResult>,
        surface_wear_score: f64,
        evidence_snapshot_paths: Option<HashMap<String, String>>,
    ) -> (DispositionResult, AuditEvidenceCertificate) {
        let msrp = sku_profile.msrp;
        let mut deductions = Vec::new();

        let has_swap_fraud = swap_result.map_or(false, |r| {
            r.is_swap_detected && r.identity_status == IdentityMatchStatus::ConfirmedSwap
        });
        let has_serial_fraud = serial_result.map_or(false, |r| {
            !r.is_match && r.match_status == SerialMatchStatus::Mismatch
        });
        let is_hard_fraud = has_swap_fraud || has_serial_fraud;

        let condition_grade = Self::determine_condition_grade(surface_wear_score, is_hard_fraud);
        let fraud_flags = Self::evaluate_fraud_flags(
            swap_result,
            serial_result,
            kitting_result,
            condition_grade,
        );

        let (action, total_deductions, net_refund, summary) = if is_hard_fraud {
            deductions.push(ItemizedDeduction {
                reason: "Return Rejected: Fraud Flagged (Bait-and-Switch or Serial Mismatch)".to_string(),
                category: "fraud_forfeit".to_string(),
                amount: msrp,
            });
            let flag_str = fraud_flags
                .iter()
                .map(|f| format!("{:?}", f))
                .collect::<Vec<_>>()
                .join(", ");
            (
                DispositionAction::RejectFraudHold,
                msrp,
                0.0,
                format!(
                    "REJECTED - FRAUD DETECTED: Return {} rejected due to critical anomalies ({}). Refund withheld in full ($0.00).",
                    rma_number, flag_str
                ),
            )
        } else {
            // Standard restocking fee
            if sku_profile.restocking_fee_rate > 0.0 {
                let fee = ((msrp * sku_profile.restocking_fee_rate) * 100.0).round() / 100.0;
                deductions.push(ItemizedDeduction {
                    reason: format!(
                        "Standard Open-Box Restocking Fee ({}%)",
                        (sku_profile.restocking_fee_rate * 100.0).round() as u32
                    ),
                    category: "restocking_fee".to_string(),
                    amount: fee,
                });
            }

            // Missing accessories
            if let Some(kit) = kitting_result {
                for item in &kit.item_audits {
                    if item.status != AccessoryStatus::PresentOem && item.deduction_fee > 0.0 {
                        deductions.push(ItemizedDeduction {
                            reason: format!("Missing OEM Accessory: {}", item.name),
                            category: "accessory".to_string(),
                            amount: item.deduction_fee,
                        });
                    }
                }
            }

            // Cosmetic wear penalty
            let wear_rate = match condition_grade {
                ConditionGrade::GradeA => 0.0,
                ConditionGrade::GradeB => 0.10,
                ConditionGrade::GradeC => 0.25,
                ConditionGrade::FraudHold => 1.0,
            };
            if wear_rate > 0.0 {
                let wear_fee = ((msrp * wear_rate) * 100.0).round() / 100.0;
                deductions.push(ItemizedDeduction {
                    reason: format!("Cosmetic Surface Wear Penalty: {:?}", condition_grade),
                    category: "cosmetic_wear".to_string(),
                    amount: wear_fee,
                });
            }

            let sum_deductions: f64 = deductions.iter().map(|d| d.amount).sum();
            let total = (sum_deductions.min(msrp) * 100.0).round() / 100.0;
            let refund = ((msrp - total).max(0.0) * 100.0).round() / 100.0;

            let is_suspected_swap = swap_result.map_or(false, |r| {
                r.identity_status == IdentityMatchStatus::SuspectedSwap
            });
            let has_incomplete_evidence = fraud_flags.contains(&FraudFlag::IncompleteEvidence);

            if has_incomplete_evidence {
                (
                    DispositionAction::ManualSecondaryReview,
                    total,
                    0.0,
                    format!(
                        "MANUAL REVIEW REQUIRED: Return {} has incomplete or unverified inspection evidence. Automated refund withheld pending review.",
                        rma_number
                    ),
                )
            } else if is_suspected_swap {
                (
                    DispositionAction::ManualSecondaryReview,
                    total,
                    refund,
                    format!(
                        "MANUAL REVIEW REQUIRED: Return {} exhibits visual ambiguity in swap detection. Tentative refund: ${:.2}.",
                        rma_number, refund
                    ),
                )
            } else if total == 0.0 {
                (
                    DispositionAction::FullRefund,
                    0.0,
                    msrp,
                    format!(
                        "FULL REFUND APPROVED: Return {} is pristine Grade A with complete accessories (${:.2}).",
                        rma_number, msrp
                    ),
                )
            } else {
                (
                    DispositionAction::PartialRefund,
                    total,
                    refund,
                    format!(
                        "PARTIAL REFUND APPROVED: Return {} net refund ${:.2} after ${:.2} in itemized deductions.",
                        rma_number, refund, total
                    ),
                )
            }
        };

        // Deterministic SHA-256 dispute certificate sealing
        let payload = serde_json::json!({
            "rma_number": rma_number,
            "sku_id": sku_profile.sku_id,
            "action": format!("{:?}", action),
            "condition_grade": format!("{:?}", condition_grade),
            "original_msrp": msrp,
            "net_refund": net_refund,
            "total_deductions": total_deductions,
            "fraud_flags": fraud_flags.iter().map(|f| format!("{:?}", f)).collect::<Vec<_>>(),
            "swap_cosine": swap_result.map(|r| r.cosine_similarity),
            "serial_matched": serial_result.map(|r| r.is_match),
            "missing_accessories_count": kitting_result.map_or(0, |r| r.missing_items_count),
            "timestamp": Utc::now().to_rfc3339(),
        });

        let mut hasher = Sha256::new();
        hasher.update(payload.to_string().as_bytes());
        let evidence_hash = format!("{:x}", hasher.finalize());

        let narrative = format!(
            "ReturnShield AI Edge Gateway Audit for RMA {} (SKU: {}, MSRP: ${:.2}).\n\
            Decision: {:?} | Condition Grade: {:?}.\n\
            Serial Verification: {}.\n\
            Visual Identity (DINOv2): {}.\n\
            Missing Accessories: {} item(s).\n\
            Itemized Deductions Applied: ${:.2}. Net Disbursed: ${:.2}.\n\
            Cryptographic Signature (SHA-256): {}",
            rma_number,
            sku_profile.sku_id,
            msrp,
            action,
            condition_grade,
            if serial_result.map_or(false, |r| r.is_match) { "MATCH" } else { "MISMATCH/UNREADABLE" },
            if swap_result.map_or(true, |r| !r.is_swap_detected) { "OEM VERIFIED" } else { "SWAP DETECTED" },
            kitting_result.map_or(0, |r| r.missing_items_count),
            total_deductions,
            net_refund,
            evidence_hash
        );

        let disposition = DispositionResult {
            action,
            condition_grade,
            original_msrp: msrp,
            total_deductions,
            net_refund_amount: net_refund,
            deduction_breakdown: deductions,
            recommendation_summary: summary,
        };

        let certificate = AuditEvidenceCertificate {
            certificate_id: format!("CERT-{}", uuid::Uuid::new_v4().simple().to_string()[..12].to_uppercase()),
            rma_number: rma_number.to_string(),
            sku_id: sku_profile.sku_id.clone(),
            timestamp: Utc::now(),
            disposition_action: action,
            condition_grade,
            fraud_flags,
            evidence_hash,
            evidence_snapshot_paths: evidence_snapshot_paths.unwrap_or_default(),
            audit_narrative: narrative,
            dispute_ready_payload: payload,
        };

        (disposition, certificate)
    }
}

