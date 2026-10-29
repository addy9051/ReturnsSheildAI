"""End-to-End Integration and SLA Test Suite for ReturnTriagePipeline (Phase 3).

Validates the complete warehouse return inspection workflow across all 5 standard scenarios:
1. Clean Return (Authentic, matching serials, complete BOM) -> FULL_REFUND
2. Bait-and-Switch Fraud (Swapped chassis visual representation) -> REJECT_FRAUD_HOLD
3. Serial Mismatch Fraud (Box label does not match chassis engraving) -> REJECT_FRAUD_HOLD
4. Missing BOM Accessories (Customer withheld cables/charger) -> PARTIAL_REFUND
5. Heavy Cosmetic Wear (Deep surface gouges/scratches) -> PARTIAL_REFUND with condition penalty

Also enforces the warehouse station SLA: warm inspection must complete well under 3.0 seconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import pytest

from core.catalog_manager import CatalogManager
from core.disposition_engine import DispositionEngine
from core.kitting_auditor import KittingAuditor
from core.pipeline import ReturnTriagePipeline
from core.schemas import (
    AccessoryStatus,
    ConditionGrade,
    DispositionAction,
    FraudFlag,
    IdentityMatchStatus,
    ReturnInspectionRequest,
    ReturnInspectionResult,
)
from core.serial_verifier import SerialVerifier
from core.swap_detector import SwapDetector
from tests.synthetic_data import MockReturnCase, generate_synthetic_return_dataset


@pytest.fixture(scope="module")
def synthetic_dataset(tmp_path_factory) -> Dict[str, MockReturnCase]:
    """Generates synthetic test cases once for the module."""
    data_dir = tmp_path_factory.mktemp("pipeline_synthetic_data")
    return generate_synthetic_return_dataset(data_dir)


@pytest.fixture(scope="module")
def e2e_pipeline(synthetic_dataset: Dict[str, MockReturnCase], tmp_path_factory) -> ReturnTriagePipeline:
    """Pre-configures a ReturnTriagePipeline with synthetic and reference SKUs."""
    catalog_dir = tmp_path_factory.mktemp("pipeline_catalog")
    catalog_mgr = CatalogManager(storage_dir=catalog_dir)

    # Use first case's chassis crop as reference for the SKU
    case_clean = synthetic_dataset["CASE-01-CLEAN"]
    sku_id = case_clean.sku_id

    # Onboard the test SKU into the catalog
    from core.schemas import BOMItem, SKUProfile
    test_bom = [
        BOMItem(item_id="cable-usbc", name="USB-C Charging Cable", category="cable", deduction_fee=15.0),
        BOMItem(item_id="case-travel", name="Hard Protective Case", category="case", deduction_fee=35.0),
        BOMItem(item_id="doc-manual", name="Quick Start & Warranty Guide", category="manual", deduction_fee=5.0),
    ]
    profile = SKUProfile(
        sku_id=sku_id,
        title="Test Over-Ear Wireless Headphones",
        category="Audio",
        msrp=399.99,
        restocking_fee_rate=0.0,
        bom=test_bom,
        expected_serial_regex=r"^SN-DEV-[A-Z0-9]{4,8}$",
    )
    catalog_mgr.onboard_sku(profile, reference_images=[case_clean.chassis_crop_path])

    # Initialize full pipeline
    pipeline = ReturnTriagePipeline(catalog_manager=catalog_mgr)

    # Warmup pass to prime PyTorch CUDA and ONNX Runtime execution graphs
    warmup_req = ReturnInspectionRequest(
        rma_number="RMA-WARMUP-00",
        sku_id=sku_id,
        box_barcode_raw="SN-DEV-WARMUP",
        chassis_crop_path=str(case_clean.chassis_crop_path),
        serial_plate_crop_path=str(case_clean.serial_plate_crop_path),
        tray_crop_path=str(case_clean.tray_crop_path),
    )
    pipeline.inspect(warmup_req)

    return pipeline


def test_pipeline_e2e_clean_return(
    e2e_pipeline: ReturnTriagePipeline,
    synthetic_dataset: Dict[str, MockReturnCase],
):
    """Scenario 1: Authentic, pristine return with matching serials and complete kit."""
    case = synthetic_dataset["CASE-01-CLEAN"]

    request = ReturnInspectionRequest(
        rma_number=case.rma_number,
        sku_id=case.sku_id,
        overhead_image_path=str(case.overhead_image_path),
        chassis_crop_path=str(case.chassis_crop_path),
        serial_plate_crop_path=str(case.serial_plate_crop_path),
        box_barcode_raw=case.box_barcode,
        tray_crop_path=str(case.tray_crop_path),
        operator_id="OPERATOR-TEST",
        bench_id="BENCH-BAY-01",
    )

    result = e2e_pipeline.inspect(request)

    assert result.rma_number == case.rma_number
    assert result.is_fraud is False
    assert result.disposition.action == DispositionAction.FULL_REFUND
    assert result.disposition.condition_grade == ConditionGrade.GRADE_A
    assert result.disposition.net_refund_amount == 399.99
    assert result.disposition.total_deductions == 0.0
    assert result.certificate is not None
    assert len(result.certificate.evidence_hash) == 64
    assert result.processing_time_ms < 3000.0, f"SLA violated: took {result.processing_time_ms}ms"


def test_pipeline_e2e_bait_and_switch(
    e2e_pipeline: ReturnTriagePipeline,
    synthetic_dataset: Dict[str, MockReturnCase],
):
    """Scenario 2: Fraudster swapped device chassis with older generation / cheap knockoff."""
    case = synthetic_dataset["CASE-02-SWAP"]

    request = ReturnInspectionRequest(
        rma_number=case.rma_number,
        sku_id=case.sku_id,
        overhead_image_path=str(case.overhead_image_path),
        chassis_crop_path=str(case.chassis_crop_path),
        serial_plate_crop_path=str(case.serial_plate_crop_path),
        box_barcode_raw=case.box_barcode,
        tray_crop_path=str(case.tray_crop_path),
    )

    result = e2e_pipeline.inspect(request)

    assert result.is_fraud is True
    assert result.disposition.action == DispositionAction.REJECT_FRAUD_HOLD
    assert result.disposition.net_refund_amount == 0.0
    assert FraudFlag.BAIT_AND_SWITCH in result.fraud_flags
    assert result.swap_result.is_swap_detected is True
    assert result.processing_time_ms < 3000.0


def test_pipeline_e2e_serial_mismatch(
    e2e_pipeline: ReturnTriagePipeline,
    synthetic_dataset: Dict[str, MockReturnCase],
):
    """Scenario 3: Box barcode does not match chassis serial number (stolen unit swap)."""
    case = synthetic_dataset["CASE-03-SERIAL-MISMATCH"]

    request = ReturnInspectionRequest(
        rma_number=case.rma_number,
        sku_id=case.sku_id,
        overhead_image_path=str(case.overhead_image_path),
        chassis_crop_path=str(case.chassis_crop_path),
        serial_plate_crop_path=str(case.serial_plate_crop_path),
        box_barcode_raw=case.box_barcode,
        tray_crop_path=str(case.tray_crop_path),
    )

    result = e2e_pipeline.inspect(request)

    assert result.is_fraud is True
    assert result.disposition.action == DispositionAction.REJECT_FRAUD_HOLD
    assert result.disposition.net_refund_amount == 0.0
    assert FraudFlag.SERIAL_MISMATCH in result.fraud_flags
    assert result.serial_result.is_match is False
    assert result.processing_time_ms < 3000.0


def test_pipeline_e2e_missing_accessories(
    e2e_pipeline: ReturnTriagePipeline,
    synthetic_dataset: Dict[str, MockReturnCase],
):
    """Scenario 4: Customer kept charging cables and manuals, returning partial kit."""
    case = synthetic_dataset["CASE-04-MISSING-ACCESSORIES"]

    request = ReturnInspectionRequest(
        rma_number=case.rma_number,
        sku_id=case.sku_id,
        overhead_image_path=str(case.overhead_image_path),
        chassis_crop_path=str(case.chassis_crop_path),
        serial_plate_crop_path=str(case.serial_plate_crop_path),
        box_barcode_raw=case.box_barcode,
        tray_crop_path=str(case.tray_crop_path),
    )

    result = e2e_pipeline.inspect(request)

    assert result.is_fraud is False
    assert result.disposition.action == DispositionAction.PARTIAL_REFUND
    assert FraudFlag.MISSING_ACCESSORY in result.fraud_flags
    assert result.kitting_result.all_items_present is False
    assert result.disposition.total_deductions > 0.0
    assert result.disposition.net_refund_amount < 399.99
    assert result.processing_time_ms < 3000.0


def test_pipeline_e2e_cosmetic_wear(
    e2e_pipeline: ReturnTriagePipeline,
    synthetic_dataset: Dict[str, MockReturnCase],
):
    """Scenario 5: Authentic return with visible scratches/scuffs triggering wear deduction."""
    case = synthetic_dataset["CASE-05-COSMETIC-WEAR"]

    request = ReturnInspectionRequest(
        rma_number=case.rma_number,
        sku_id=case.sku_id,
        overhead_image_path=str(case.overhead_image_path),
        chassis_crop_path=str(case.chassis_crop_path),
        serial_plate_crop_path=str(case.serial_plate_crop_path),
        box_barcode_raw=case.box_barcode,
        tray_crop_path=str(case.tray_crop_path),
    )

    result = e2e_pipeline.inspect(request)

    assert result.is_fraud is False
    assert result.swap_result.identity_status in [IdentityMatchStatus.VERIFIED_OEM, IdentityMatchStatus.SUSPECTED_SWAP]
    assert result.serial_result.is_match is True
    assert result.disposition.condition_grade == ConditionGrade.GRADE_C
    assert FraudFlag.PHYSICAL_DAMAGE in result.fraud_flags
    assert result.processing_time_ms < 3000.0


def test_pipeline_missing_chassis_crop_graceful_fallback(
    e2e_pipeline: ReturnTriagePipeline,
    synthetic_dataset: Dict[str, MockReturnCase],
):
    """Tests that pipeline gracefully degrades if optional ROI crops are omitted."""
    case = synthetic_dataset["CASE-01-CLEAN"]

    # Only supply overhead snapshot and box barcode
    request = ReturnInspectionRequest(
        rma_number="RMA-DEGRADED-TEST",
        sku_id=case.sku_id,
        overhead_image_path=str(case.overhead_image_path),
        box_barcode_raw=case.box_barcode,
    )

    result = e2e_pipeline.inspect(request)
    assert result.rma_number == "RMA-DEGRADED-TEST"
    assert result.disposition is not None
    assert result.certificate is not None
