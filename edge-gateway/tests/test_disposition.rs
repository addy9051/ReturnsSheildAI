use returnshield_edge::models::*;
use returnshield_edge::verification::DispositionEngine;

fn sample_profile() -> SkuProfile {
    SkuProfile {
        sku_id: "SKU-TEST-400".to_string(),
        title: "Test ANC Headphones".to_string(),
        category: "Audio".to_string(),
        msrp: 399.99,
        restocking_fee_rate: 0.10,
        expected_serial_regex: None,
        reference_image_paths: Vec::new(),
        bom: vec![
            BOMItem {
                item_id: "cable-usbc".to_string(),
                name: "USB-C Cable".to_string(),
                category: "cable".to_string(),
                required_qty: 1,
                deduction_fee: 25.0,
                expected_tray_slot: None,
                is_critical: false,
            },
            BOMItem {
                item_id: "case-travel".to_string(),
                name: "Travel Case".to_string(),
                category: "case".to_string(),
                required_qty: 1,
                deduction_fee: 40.0,
                expected_tray_slot: None,
                is_critical: false,
            },
        ],
        similarity_threshold: 0.85,
        created_at: Some(chrono::Utc::now().to_rfc3339()),
        updated_at: Some(chrono::Utc::now().to_rfc3339()),
        metadata: std::collections::HashMap::new(),
    }
}

#[test]
fn test_clean_return_disposition() {
    let mut profile = sample_profile();
    profile.restocking_fee_rate = 0.0;

    let swap_res = SwapDetectionResult {
        identity_status: IdentityMatchStatus::VerifiedOem,
        cosine_similarity: 0.985,
        threshold: 0.85,
        is_swap_detected: false,
        explanation: "Verified OEM".to_string(),
        spatial_diff_score: None,
        heatmap_path: None,
    };

    let serial_res = SerialVerificationResult {
        box_barcode: Some("SN-12345".to_string()),
        chassis_serial_raw: Some("SN-12345".to_string()),
        chassis_serial_normalized: Some("SN12345".to_string()),
        match_status: SerialMatchStatus::ExactMatch,
        is_match: true,
        similarity_ratio: 1.0,
        ocr_confidence: 0.99,
        notes: "Exact match".to_string(),
    };

    let (disp, cert) = DispositionEngine::calculate_disposition(
        &profile,
        "RMA-RUST-01",
        Some(&swap_res),
        Some(&serial_res),
        None,
        0.0,
        None,
    );

    assert_eq!(disp.action, DispositionAction::FullRefund);
    assert_eq!(disp.condition_grade, ConditionGrade::GradeA);
    assert_eq!(disp.net_refund_amount, 399.99);
    assert_eq!(disp.total_deductions, 0.0);
    assert_eq!(cert.fraud_flags, vec![FraudFlag::Clean]);
    assert_eq!(cert.evidence_hash.len(), 64);
}

#[test]
fn test_bait_and_switch_disposition() {
    let profile = sample_profile();

    let swap_res = SwapDetectionResult {
        identity_status: IdentityMatchStatus::ConfirmedSwap,
        cosine_similarity: 0.58,
        threshold: 0.85,
        is_swap_detected: true,
        explanation: "Confirmed model swap".to_string(),
        spatial_diff_score: Some(0.85),
        heatmap_path: None,
    };

    let (disp, cert) = DispositionEngine::calculate_disposition(
        &profile,
        "RMA-RUST-SWAP",
        Some(&swap_res),
        None,
        None,
        0.0,
        None,
    );

    assert_eq!(disp.action, DispositionAction::RejectFraudHold);
    assert_eq!(disp.condition_grade, ConditionGrade::FraudHold);
    assert_eq!(disp.net_refund_amount, 0.0);
    assert_eq!(disp.total_deductions, 399.99);
    assert!(cert.fraud_flags.contains(&FraudFlag::BaitAndSwitch));
}

#[test]
fn test_partial_refund_missing_items_and_wear() {
    let profile = sample_profile(); // 10% restocking fee = $40.00

    let kitting_res = KittingAuditResult {
        expected_items_count: 2,
        detected_items_count: 1,
        missing_items_count: 1,
        all_items_present: false,
        item_audits: vec![
            KittedItemAudit {
                bom_item_id: "cable-usbc".to_string(),
                name: "USB-C Cable".to_string(),
                status: AccessoryStatus::PresentOem,
                confidence: 0.98,
                deduction_fee: 0.0,
            },
            KittedItemAudit {
                bom_item_id: "case-travel".to_string(),
                name: "Travel Case".to_string(),
                status: AccessoryStatus::Missing,
                confidence: 0.95,
                deduction_fee: 40.0,
            },
        ],
        total_accessory_deduction: 40.0,
    };

    // Wear score 0.20 -> Grade B -> 10% penalty = $40.00
    let (disp, cert) = DispositionEngine::calculate_disposition(
        &profile,
        "RMA-RUST-PARTIAL",
        None,
        None,
        Some(&kitting_res),
        0.20,
        None,
    );

    assert_eq!(disp.action, DispositionAction::PartialRefund);
    assert_eq!(disp.condition_grade, ConditionGrade::GradeB);
    // Deductions: Restocking $40.00 + Missing Case $40.00 + Wear $40.00 = $120.00
    assert_eq!(disp.total_deductions, 120.0);
    assert_eq!(disp.net_refund_amount, 279.99);
    assert!(cert.fraud_flags.contains(&FraudFlag::MissingAccessory));
}
