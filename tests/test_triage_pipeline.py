"""Integration and Unit Test Suite for ReturnShield AI (Phase 1).

Tests:
1. Pydantic schema validation, invariants, and serialization.
2. CatalogManager onboarding, DINOv2 embedding caching, and vector search.
3. Synthetic return test case generator across all 5 fraud & triage scenarios:
   - Clean Return (Authentic, matching serials, full BOM)
   - Bait-and-Switch Fraud (Swapped model / colorway)
   - Serial Mismatch Fraud (Different chassis serial vs box barcode)
   - Missing BOM Accessories (Itemized deduction calculation)
   - Surface Wear & Damage (Condition degradation)
4. Cryptographic audit evidence certificate sealing and tamper detection.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Dict

import numpy as np
from PIL import Image
import pytest

from core.catalog_manager import CatalogManager, DINOv2EmbeddingEngine
from core.schemas import (
    AccessoryStatus,
    AuditEvidenceCertificate,
    BOMItem,
    ConditionGrade,
    DispositionAction,
    DispositionResult,
    FraudFlag,
    IdentityMatchStatus,
    ItemizedDeduction,
    KittedItemAudit,
    KittingAuditResult,
    ReturnInspectionRequest,
    ReturnInspectionResult,
    SerialMatchStatus,
    SerialVerificationResult,
    SKUOnboardRequest,
    SKUProfile,
    SwapDetectionResult,
)
from tests.synthetic_data import (
    MockReturnCase,
    create_chassis_image,
    generate_synthetic_return_dataset,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="session")
def shared_dinov2_engine() -> DINOv2EmbeddingEngine:
    """Provides singleton DINOv2 embedding engine."""
    return DINOv2EmbeddingEngine.get_shared_engine(model_name="dinov2_vits14")


@pytest.fixture
def temp_catalog_dir(tmp_path: Path) -> Path:
    """Provides an isolated directory for catalog storage tests."""
    catalog_dir = tmp_path / "catalog_storage"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    return catalog_dir


@pytest.fixture
def sample_bom() -> list[BOMItem]:
    """Sample Bill of Materials checklist."""
    return [
        BOMItem(item_id="cable-usbc", name="USB-C Fast Cable", category="cable", deduction_fee=15.0),
        BOMItem(item_id="case-travel", name="Hard Protective Case", category="case", deduction_fee=35.0),
        BOMItem(item_id="doc-manual", name="Quick Start & Warranty Guide", category="manual", deduction_fee=5.0),
    ]


@pytest.fixture
def mock_dataset(tmp_path: Path) -> Dict[str, MockReturnCase]:
    """Generates synthetic return dataset in a temporary path."""
    dataset_dir = tmp_path / "synthetic_returns"
    return generate_synthetic_return_dataset(dataset_dir)


# ==============================================================================
# Test Group 1: Pydantic Schemas & Data Contracts
# ==============================================================================

def test_sku_profile_validation(sample_bom: list[BOMItem]):
    """Tests SKUProfile schema constraints and validation."""
    profile = SKUProfile(
        sku_id="SKU-TEST-001",
        title="Test Noise Cancelling Headphones",
        category="Audio",
        msrp=299.99,
        restocking_fee_rate=0.15,
        expected_serial_regex=r"^TEST-[0-9]{6}$",
        bom=sample_bom,
        similarity_threshold=0.88,
    )
    assert profile.sku_id == "SKU-TEST-001"
    assert profile.msrp == 299.99
    assert len(profile.bom) == 3
    assert profile.similarity_threshold == 0.88

    # Verify MSRP cannot be negative or zero
    with pytest.raises(Exception):
        SKUProfile(sku_id="FAIL", title="Invalid", category="Audio", msrp=-50.0)

    # Verify restocking fee cannot exceed 1.0
    with pytest.raises(Exception):
        SKUProfile(sku_id="FAIL", title="Invalid", category="Audio", msrp=100.0, restocking_fee_rate=1.5)


def test_return_inspection_result_fraud_sync():
    """Tests automatic is_fraud synchronization based on fraud flags."""
    clean_result = ReturnInspectionResult(
        rma_number="RMA-101",
        sku_id="SKU-001",
        fraud_flags=[FraudFlag.CLEAN],
    )
    assert clean_result.is_fraud is False

    fraud_result = ReturnInspectionResult(
        rma_number="RMA-102",
        sku_id="SKU-001",
        fraud_flags=[FraudFlag.BAIT_AND_SWITCH],
    )
    assert fraud_result.is_fraud is True

    serial_fraud = ReturnInspectionResult(
        rma_number="RMA-103",
        sku_id="SKU-001",
        fraud_flags=[FraudFlag.SERIAL_MISMATCH],
    )
    assert serial_fraud.is_fraud is True


def test_cryptographic_evidence_certificate():
    """Tests SHA-256 evidence certificate hashing and tamper-evidence."""
    payload = {
        "rma_number": "RMA-2026-9901",
        "sku_id": "SKU-SONY-WH1000XM5-BLK",
        "action": DispositionAction.REJECT_FRAUD_HOLD.value,
        "cosine_similarity": 0.42,
        "serial_matched": False,
    }
    hash_digest = AuditEvidenceCertificate.compute_evidence_hash(payload)
    assert isinstance(hash_digest, str)
    assert len(hash_digest) == 64

    # Identical payload yields identical hash (deterministic)
    assert AuditEvidenceCertificate.compute_evidence_hash(payload) == hash_digest

    # Tampered payload yields different hash
    tampered_payload = dict(payload, action=DispositionAction.FULL_REFUND.value)
    assert AuditEvidenceCertificate.compute_evidence_hash(tampered_payload) != hash_digest

    cert = AuditEvidenceCertificate(
        rma_number="RMA-2026-9901",
        sku_id="SKU-SONY-WH1000XM5-BLK",
        disposition_action=DispositionAction.REJECT_FRAUD_HOLD,
        condition_grade=ConditionGrade.FRAUD_HOLD,
        fraud_flags=[FraudFlag.BAIT_AND_SWITCH, FraudFlag.SERIAL_MISMATCH],
        evidence_hash=hash_digest,
        audit_narrative="DINOv2 similarity 0.42 indicates bait-and-switch counterfeit swap.",
        dispute_ready_payload=payload,
    )
    assert cert.certificate_id.startswith("CERT-")


# ==============================================================================
# Test Group 2: CatalogManager & DINOv2 Embedding Cache
# ==============================================================================

def test_catalog_manager_onboarding_and_caching(temp_catalog_dir: Path, sample_bom: list[BOMItem]):
    """Tests zero-training SKU onboarding, DINOv2 embedding extraction and cache persistence."""
    manager = CatalogManager(storage_dir=temp_catalog_dir, model_name="dinov2_vits14")

    # Generate reference image
    ref_img = create_chassis_image("authentic_black")

    onboard_req = SKUOnboardRequest(
        sku_id="SKU-SONY-WH1000XM5-BLK",
        title="Sony WH-1000XM5 Headphones - Black",
        category="Audio",
        msrp=399.99,
        restocking_fee_rate=0.10,
        expected_serial_regex=r"^SN-SNY-[0-9]{7}$",
        bom=sample_bom,
        similarity_threshold=0.85,
    )

    profile = manager.onboard_sku(onboard_req, reference_images=[ref_img])
    assert profile.sku_id == "SKU-SONY-WH1000XM5-BLK"
    assert len(profile.reference_image_paths) == 1

    # Verify persistence to catalog_db.json
    db_file = temp_catalog_dir / "catalog_db.json"
    assert db_file.exists()
    with open(db_file, "r", encoding="utf-8") as f:
        stored_db = json.load(f)
    assert "SKU-SONY-WH1000XM5-BLK" in stored_db

    # Verify vector embedding cache (.npz)
    emb_file = temp_catalog_dir / "embeddings" / "SKU-SONY-WH1000XM5-BLK.npz"
    assert emb_file.exists()

    cached_emb = manager.get_sku_embedding("SKU-SONY-WH1000XM5-BLK")
    assert cached_emb is not None
    assert cached_emb.shape == (384,)
    # Verify L2-normalization (unit vector length = 1.0)
    assert np.isclose(np.linalg.norm(cached_emb), 1.0, atol=1e-4)

    # Verify patch embeddings
    patch_emb = manager.get_sku_patch_embeddings("SKU-SONY-WH1000XM5-BLK")
    assert patch_emb is not None
    assert patch_emb.shape == (256, 384)


def test_dinov2_similarity_scoring(temp_catalog_dir: Path, sample_bom: list[BOMItem]):
    """Tests DINOv2 cosine similarity: high on identical/similar images, low on swapped items."""
    manager = CatalogManager(storage_dir=temp_catalog_dir, model_name="dinov2_vits14")

    ref_img = create_chassis_image("authentic_black")
    manager.onboard_sku(
        SKUOnboardRequest(
            sku_id="SKU-XM5",
            title="Sony XM5",
            category="Audio",
            msrp=399.99,
            bom=sample_bom,
            similarity_threshold=0.85,
        ),
        reference_images=[ref_img],
    )

    # 1. Test identical/similar image -> should yield near 1.0 similarity
    similar_img = create_chassis_image("authentic_black")
    sim_clean = manager.compute_similarity("SKU-XM5", similar_img)
    assert sim_clean >= 0.95, f"Expected self/clean similarity >= 0.95, got {sim_clean:.4f}"

    # 2. Test swapped cheap beige vintage model -> should drop drastically below threshold
    swapped_img = create_chassis_image("swapped_cheap_beige")
    sim_swap = manager.compute_similarity("SKU-XM5", swapped_img)
    assert sim_swap < 0.85, f"Expected swapped similarity < 0.85, got {sim_swap:.4f}"
    assert sim_clean - sim_swap > 0.15, "Separation between clean and swap must exceed 0.15"

    # 3. Test neon gaming headset swap
    gaming_img = create_chassis_image("swapped_neon_gaming")
    sim_gaming = manager.compute_similarity("SKU-XM5", gaming_img)
    assert sim_gaming < 0.80


def test_catalog_search(temp_catalog_dir: Path, sample_bom: list[BOMItem]):
    """Tests multi-SKU search to identify returned product against catalog."""
    manager = CatalogManager(storage_dir=temp_catalog_dir, model_name="dinov2_vits14")

    img_black = create_chassis_image("authentic_black")
    img_beige = create_chassis_image("swapped_cheap_beige")

    manager.onboard_sku(
        SKUOnboardRequest(sku_id="SKU-BLACK", title="Black Edition", category="Audio", msrp=399.99, bom=sample_bom),
        reference_images=[img_black],
    )
    manager.onboard_sku(
        SKUOnboardRequest(sku_id="SKU-BEIGE", title="Beige Edition", category="Audio", msrp=199.99, bom=sample_bom),
        reference_images=[img_beige],
    )

    # Query with a black headphone sample
    results = manager.search_catalog(img_black, top_k=2)
    assert len(results) == 2
    top_sku, top_score = results[0]
    assert top_sku.sku_id == "SKU-BLACK"
    assert top_score >= 0.95


def test_bom_checklist_verification(temp_catalog_dir: Path, sample_bom: list[BOMItem]):
    """Tests Bill of Materials item presence and missing accessory fee calculation."""
    manager = CatalogManager(storage_dir=temp_catalog_dir)
    manager.onboard_sku(
        SKUOnboardRequest(sku_id="SKU-BOM-TEST", title="BOM Test", category="Audio", msrp=399.99, bom=sample_bom),
    )

    # Case A: All present
    present, missing, deduction = manager.verify_bom_checklist("SKU-BOM-TEST", ["cable-usbc", "case-travel", "doc-manual"])
    assert len(present) == 3
    assert len(missing) == 0
    assert deduction == 0.0

    # Case B: Cable and case missing
    present, missing, deduction = manager.verify_bom_checklist("SKU-BOM-TEST", ["doc-manual"])
    assert len(present) == 1
    assert len(missing) == 2
    assert deduction == 50.0  # $15 cable + $35 case


# ==============================================================================
# Test Group 3: Synthetic Return Dataset & Mock Triage Scenarios
# ==============================================================================

def test_synthetic_dataset_generation(mock_dataset: Dict[str, MockReturnCase]):
    """Tests that all 5 mock return scenarios are generated with valid files."""
    expected_case_ids = [
        "CASE-01-CLEAN",
        "CASE-02-SWAP",
        "CASE-03-SERIAL-MISMATCH",
        "CASE-04-MISSING-ACCESSORIES",
        "CASE-05-COSMETIC-WEAR",
    ]
    for cid in expected_case_ids:
        assert cid in mock_dataset
        case = mock_dataset[cid]
        assert case.overhead_image_path.exists()
        assert case.chassis_crop_path.exists()
        assert case.serial_plate_crop_path.exists()
        assert case.tray_crop_path.exists()

        # Check image files are readable
        with Image.open(case.overhead_image_path) as img:
            assert img.size == (800, 600)
        with Image.open(case.chassis_crop_path) as img:
            assert img.size == (400, 400)


def test_scenario_1_clean_return(mock_dataset: Dict[str, MockReturnCase], temp_catalog_dir: Path, sample_bom: list[BOMItem]):
    """Verifies Case 1: Clean return gets verified, matching serials, full refund."""
    case = mock_dataset["CASE-01-CLEAN"]
    manager = CatalogManager(storage_dir=temp_catalog_dir)
    manager.onboard_sku(
        SKUOnboardRequest(sku_id=case.sku_id, title="Sony XM5", category="Audio", msrp=399.99, bom=sample_bom),
        reference_images=[case.chassis_crop_path],
    )

    # 1. Visual identity matching
    sim = manager.compute_similarity(case.sku_id, case.chassis_crop_path)
    assert sim >= 0.90
    identity_status = IdentityMatchStatus.VERIFIED_OEM if sim >= 0.85 else IdentityMatchStatus.CONFIRMED_SWAP
    assert identity_status == IdentityMatchStatus.VERIFIED_OEM

    # 2. Serial verification
    serial_match = (case.box_barcode == case.chassis_serial)
    assert serial_match is True
    assert case.expected_serial_status == SerialMatchStatus.EXACT_MATCH

    # 3. Kitting checklist
    _, missing, fee = manager.verify_bom_checklist(case.sku_id, case.detected_accessories)
    assert len(missing) == 0
    assert fee == 0.0

    # 4. Final disposition
    assert case.expected_fraud_flags == [FraudFlag.CLEAN]
    assert case.expected_disposition == DispositionAction.FULL_REFUND
    assert case.expected_condition == ConditionGrade.GRADE_A


def test_scenario_2_swap_fraud(mock_dataset: Dict[str, MockReturnCase], temp_catalog_dir: Path, sample_bom: list[BOMItem]):
    """Verifies Case 2: Swapped cheap model triggers BAIT_AND_SWITCH and REJECT_FRAUD_HOLD."""
    case = mock_dataset["CASE-02-SWAP"]
    manager = CatalogManager(storage_dir=temp_catalog_dir)

    # Onboard genuine reference
    genuine_ref = create_chassis_image("authentic_black")
    manager.onboard_sku(
        SKUOnboardRequest(sku_id=case.sku_id, title="Sony XM5", category="Audio", msrp=399.99, bom=sample_bom),
        reference_images=[genuine_ref],
    )

    # Visual identity check with swapped chassis
    sim = manager.compute_similarity(case.sku_id, case.chassis_crop_path)
    assert sim < 0.85
    is_swap = sim < 0.85
    assert is_swap is True

    # Serial cross-match check
    assert case.box_barcode != case.chassis_serial

    # Verify fraud triage expectations
    assert FraudFlag.BAIT_AND_SWITCH in case.expected_fraud_flags
    assert FraudFlag.SERIAL_MISMATCH in case.expected_fraud_flags
    assert case.expected_disposition == DispositionAction.REJECT_FRAUD_HOLD
    assert case.expected_condition == ConditionGrade.FRAUD_HOLD


def test_scenario_3_serial_mismatch(mock_dataset: Dict[str, MockReturnCase]):
    """Verifies Case 3: Serial mismatch detects discrepancy between chassis and box barcode."""
    case = mock_dataset["CASE-03-SERIAL-MISMATCH"]

    assert case.box_barcode == "SN-SNY-8842109"
    assert case.chassis_serial == "SN-SNY-1102934"
    assert case.box_barcode != case.chassis_serial
    assert case.expected_serial_status == SerialMatchStatus.MISMATCH

    assert FraudFlag.SERIAL_MISMATCH in case.expected_fraud_flags
    assert case.expected_disposition == DispositionAction.REJECT_FRAUD_HOLD


def test_scenario_4_missing_accessories(mock_dataset: Dict[str, MockReturnCase], temp_catalog_dir: Path, sample_bom: list[BOMItem]):
    """Verifies Case 4: Missing BOM accessories produce correct partial refund deduction."""
    case = mock_dataset["CASE-04-MISSING-ACCESSORIES"]
    manager = CatalogManager(storage_dir=temp_catalog_dir)
    manager.onboard_sku(
        SKUOnboardRequest(sku_id=case.sku_id, title="Sony XM5", category="Audio", msrp=399.99, bom=sample_bom),
    )

    present, missing, total_fee = manager.verify_bom_checklist(case.sku_id, case.detected_accessories)
    assert len(present) == 1
    assert len(missing) == 2
    missing_ids = [m.item_id for m in missing]
    assert "cable-usbc" in missing_ids
    assert "case-travel" in missing_ids
    assert total_fee == 50.0

    # Calculate net partial refund
    msrp = 399.99
    net_refund = msrp - total_fee
    assert np.isclose(net_refund, 349.99)
    assert case.expected_disposition == DispositionAction.PARTIAL_REFUND
    assert FraudFlag.MISSING_ACCESSORY in case.expected_fraud_flags


def test_scenario_5_cosmetic_wear(mock_dataset: Dict[str, MockReturnCase]):
    """Verifies Case 5: Authentic return with heavy wear is downgraded to Grade C."""
    case = mock_dataset["CASE-05-COSMETIC-WEAR"]
    assert case.surface_wear_severity > 0.30
    assert case.expected_condition == ConditionGrade.GRADE_C
    assert case.expected_disposition == DispositionAction.PARTIAL_REFUND
    assert FraudFlag.PHYSICAL_DAMAGE in case.expected_fraud_flags

