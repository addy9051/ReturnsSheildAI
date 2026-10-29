"""Unit and Integration Tests for DispositionEngine (Stage 4 Financial Triage & Disputes).

Tests:
1. Condition grading logic based on surface abrasion and fraud state.
2. Clean return triage (FULL_REFUND, zero fraud flags, Grade A).
3. Hard fraud rejection on Bait-and-Switch (REJECT_FRAUD_HOLD, $0.00 refund).
4. Hard fraud rejection on Serial Number mismatch.
5. Partial refund with itemized restocking fee, missing BOM deductions, and cosmetic wear penalty.
6. Cryptographic SHA-256 Audit Evidence Certificate sealing and tamper detection.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pytest

from core.disposition_engine import DispositionEngine
from core.schemas import (
    AccessoryStatus,
    AuditEvidenceCertificate,
    BOMItem,
    ConditionGrade,
    DispositionAction,
    FraudFlag,
    IdentityMatchStatus,
    KittedItemAudit,
    KittingAuditResult,
    SerialMatchStatus,
    SerialVerificationResult,
    SKUProfile,
    SwapDetectionResult,
)


@pytest.fixture
def test_sku_profile() -> SKUProfile:
    """Standard SKU profile for $400 electronics."""
    return SKUProfile(
        sku_id="SKU-AUDIO-400",
        title="Premium Wireless ANC Headphones",
        category="Audio",
        msrp=399.99,
        restocking_fee_rate=0.0,  # 0% restocking fee for baseline clean test
        bom=[
            BOMItem(item_id="cable-usbc", name="USB-C Charging Cable", category="cable", deduction_fee=25.0),
            BOMItem(item_id="case-carrying", name="Protective Travel Case", category="case", deduction_fee=40.0),
        ],
        expected_serial_regex=r"^SN-AUD-[A-Z0-9]{8}$",
    )


@pytest.fixture
def disposition_engine() -> DispositionEngine:
    return DispositionEngine()


def test_determine_condition_grade(disposition_engine: DispositionEngine):
    """Tests surface wear threshold mapping to condition grades."""
    # Mint / Like New
    assert disposition_engine.determine_condition_grade(0.02, is_fraud=False) == ConditionGrade.GRADE_A
    assert disposition_engine.determine_condition_grade(0.10, is_fraud=False) == ConditionGrade.GRADE_A

    # Minor Blemishes
    assert disposition_engine.determine_condition_grade(0.15, is_fraud=False) == ConditionGrade.GRADE_B
    assert disposition_engine.determine_condition_grade(0.30, is_fraud=False) == ConditionGrade.GRADE_B

    # Heavy Scratches / Dents
    assert disposition_engine.determine_condition_grade(0.35, is_fraud=False) == ConditionGrade.GRADE_C
    assert disposition_engine.determine_condition_grade(0.85, is_fraud=False) == ConditionGrade.GRADE_C

    # Fraud always overrides to FRAUD_HOLD regardless of physical cleanliness
    assert disposition_engine.determine_condition_grade(0.01, is_fraud=True) == ConditionGrade.FRAUD_HOLD


def test_clean_return_full_refund(
    disposition_engine: DispositionEngine,
    test_sku_profile: SKUProfile,
):
    """Tests pristine return yields FULL_REFUND with 0 deductions and Grade A."""
    swap_res = SwapDetectionResult(
        sku_id=test_sku_profile.sku_id,
        cosine_similarity=0.985,
        identity_status=IdentityMatchStatus.VERIFIED_OEM,
        is_swap_detected=False,
    )
    serial_res = SerialVerificationResult(
        box_barcode_value="SN-AUD-ABC12345",
        chassis_serial_value="SN-AUD-ABC12345",
        is_match=True,
        match_status=SerialMatchStatus.EXACT_MATCH,
        similarity_ratio=1.0,
    )
    kitting_res = KittingAuditResult(
        expected_items_count=2,
        detected_items_count=2,
        missing_items_count=0,
        all_items_present=True,
        item_audits=[
            KittedItemAudit(bom_item_id="cable-usbc", name="USB-C Cable", status=AccessoryStatus.PRESENT_OEM, deduction_fee=0.0),
            KittedItemAudit(bom_item_id="case-carrying", name="Case", status=AccessoryStatus.PRESENT_OEM, deduction_fee=0.0),
        ],
        total_accessory_deduction=0.0,
    )

    disposition, cert = disposition_engine.calculate_disposition(
        sku_profile=test_sku_profile,
        rma_number="RMA-CLEAN-001",
        swap_result=swap_res,
        serial_result=serial_res,
        kitting_result=kitting_res,
        surface_wear_score=0.02,
    )

    assert disposition.action == DispositionAction.FULL_REFUND
    assert disposition.condition_grade == ConditionGrade.GRADE_A
    assert disposition.net_refund_amount == 399.99
    assert disposition.total_deductions == 0.0
    assert cert.fraud_flags == [FraudFlag.CLEAN]
    assert len(cert.evidence_hash) == 64


def test_bait_and_switch_fraud_rejection(
    disposition_engine: DispositionEngine,
    test_sku_profile: SKUProfile,
):
    """Tests visual swap fraud triggers 100% forfeiture and REJECT_FRAUD_HOLD."""
    swap_res = SwapDetectionResult(
        sku_id=test_sku_profile.sku_id,
        cosine_similarity=0.620,
        identity_status=IdentityMatchStatus.CONFIRMED_SWAP,
        is_swap_detected=True,
    )
    serial_res = SerialVerificationResult(
        box_barcode_value="SN-AUD-ABC12345",
        chassis_serial_value="SN-AUD-ABC12345",
        is_match=True,
        match_status=SerialMatchStatus.EXACT_MATCH,
        similarity_ratio=1.0,
    )

    disposition, cert = disposition_engine.calculate_disposition(
        sku_profile=test_sku_profile,
        rma_number="RMA-SWAP-002",
        swap_result=swap_res,
        serial_result=serial_res,
    )

    assert disposition.action == DispositionAction.REJECT_FRAUD_HOLD
    assert disposition.condition_grade == ConditionGrade.FRAUD_HOLD
    assert disposition.net_refund_amount == 0.0
    assert disposition.total_deductions == 399.99
    assert FraudFlag.BAIT_AND_SWITCH in cert.fraud_flags
    assert "REJECTED - FRAUD DETECTED" in disposition.recommendation_summary


def test_serial_mismatch_fraud_rejection(
    disposition_engine: DispositionEngine,
    test_sku_profile: SKUProfile,
):
    """Tests chassis serial mismatch triggers REJECT_FRAUD_HOLD."""
    swap_res = SwapDetectionResult(
        sku_id=test_sku_profile.sku_id,
        cosine_similarity=0.970,
        identity_status=IdentityMatchStatus.VERIFIED_OEM,
        is_swap_detected=False,
    )
    serial_res = SerialVerificationResult(
        box_barcode_value="SN-AUD-ORIGINAL1",
        chassis_serial_value="SN-AUD-STOLEN999",
        is_match=False,
        match_status=SerialMatchStatus.MISMATCH,
        similarity_ratio=0.45,
    )

    disposition, cert = disposition_engine.calculate_disposition(
        sku_profile=test_sku_profile,
        rma_number="RMA-SERIAL-003",
        swap_result=swap_res,
        serial_result=serial_res,
    )

    assert disposition.action == DispositionAction.REJECT_FRAUD_HOLD
    assert disposition.condition_grade == ConditionGrade.FRAUD_HOLD
    assert disposition.net_refund_amount == 0.0
    assert disposition.total_deductions == 399.99
    assert FraudFlag.SERIAL_MISMATCH in cert.fraud_flags


def test_partial_refund_with_itemized_deductions(
    disposition_engine: DispositionEngine,
    test_sku_profile: SKUProfile,
):
    """Tests legitimate return with missing accessories and cosmetic wear."""
    # Enable 10% restocking fee on SKU profile
    test_sku_profile.restocking_fee_rate = 0.10

    swap_res = SwapDetectionResult(
        sku_id=test_sku_profile.sku_id,
        cosine_similarity=0.980,
        identity_status=IdentityMatchStatus.VERIFIED_OEM,
        is_swap_detected=False,
    )
    serial_res = SerialVerificationResult(
        box_barcode_value="SN-AUD-ABC12345",
        chassis_serial_value="SN-AUD-ABC12345",
        is_match=True,
        match_status=SerialMatchStatus.EXACT_MATCH,
        similarity_ratio=1.0,
    )
    # Missing travel case ($40.00)
    kitting_res = KittingAuditResult(
        expected_items_count=2,
        detected_items_count=1,
        missing_items_count=1,
        all_items_present=False,
        item_audits=[
            KittedItemAudit(bom_item_id="cable-usbc", name="USB-C Cable", status=AccessoryStatus.PRESENT_OEM, deduction_fee=0.0),
            KittedItemAudit(bom_item_id="case-carrying", name="Protective Travel Case", status=AccessoryStatus.MISSING, deduction_fee=40.0),
        ],
        total_accessory_deduction=40.0,
    )

    # Moderate wear (score 0.20 -> Grade B -> 10% penalty of $399.99 = $40.00)
    disposition, cert = disposition_engine.calculate_disposition(
        sku_profile=test_sku_profile,
        rma_number="RMA-PARTIAL-004",
        swap_result=swap_res,
        serial_result=serial_res,
        kitting_result=kitting_res,
        surface_wear_score=0.20,
    )

    assert disposition.action == DispositionAction.PARTIAL_REFUND
    assert disposition.condition_grade == ConditionGrade.GRADE_B
    # Restocking fee (10% = $40.00) + Missing case ($40.00) + Wear penalty (10% = $40.00) = $120.00
    expected_deductions = 40.00 + 40.00 + 40.00
    assert disposition.total_deductions == expected_deductions
    # Net refund = $399.99 - $120.00 = $279.99
    assert disposition.net_refund_amount == 279.99
    assert FraudFlag.MISSING_ACCESSORY in cert.fraud_flags


def test_cryptographic_evidence_sealing_and_tamper_detection(
    disposition_engine: DispositionEngine,
    test_sku_profile: SKUProfile,
):
    """Validates SHA-256 seal integrity and tamper sensitivity."""
    _, cert = disposition_engine.calculate_disposition(
        sku_profile=test_sku_profile,
        rma_number="RMA-SEAL-TEST",
    )

    # 1. Verify that re-hashing the payload yields the exact same signature
    recomputed_hash = AuditEvidenceCertificate.compute_evidence_hash(cert.dispute_ready_payload)
    assert cert.evidence_hash == recomputed_hash

    # 2. Tamper simulation: fraudster attempts to falsify net_refund in the payload
    tampered_payload = copy.deepcopy(cert.dispute_ready_payload)
    tampered_payload["net_refund"] = 9999.99

    tampered_hash = AuditEvidenceCertificate.compute_evidence_hash(tampered_payload)
    assert tampered_hash != cert.evidence_hash
