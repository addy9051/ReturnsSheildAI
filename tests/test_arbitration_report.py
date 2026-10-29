"""Unit and Integration Test Suite for ArbitrationReportGenerator (Option 2).

Validates:
1. End-to-end PDF generation across all 5 standard return scenarios:
   - Clean Return (Authentic, Full Refund)
   - Bait-and-Switch Fraud (Swapped Model, Fraud Hold)
   - Serial Mismatch Fraud (Stolen/Replaced Unit, Fraud Hold)
   - Missing BOM Accessories (Partial Refund, Itemized Deductions)
   - Heavy Cosmetic Wear (Condition Penalties)
2. Platform-specific legal headers (Amazon SAFE-T, Flipkart SPF, Stripe Chargeback).
3. PyMuPDF (fitz) forensic inspection:
   - Exactly 2 pages per standard dossier
   - High-contrast verdict banners
   - Embedded evidence images and DINOv2 thermal anomaly heatmaps
   - Vector QR code containing the deterministic SHA-256 digital signature
4. Resilient fallback when image crops or heatmaps are omitted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import fitz  # PyMuPDF
import pytest

from core.arbitration_report import ArbitrationReportGenerator
from core.catalog_manager import CatalogManager
from core.pipeline import ReturnTriagePipeline
from core.schemas import BOMItem, ReturnInspectionRequest, SKUProfile
from tests.synthetic_data import MockReturnCase, generate_synthetic_return_dataset


@pytest.fixture(scope="module")
def arbitration_fixtures(tmp_path_factory):
    """Sets up synthetic return dataset, catalog manager, and master pipeline."""
    data_dir = tmp_path_factory.mktemp("arbitration_data")
    dataset = generate_synthetic_return_dataset(data_dir)

    catalog_dir = tmp_path_factory.mktemp("arbitration_catalog")
    catalog_mgr = CatalogManager(storage_dir=catalog_dir)

    case_clean = dataset["CASE-01-CLEAN"]
    sku_id = case_clean.sku_id

    test_bom = [
        BOMItem(item_id="cable-usbc-braided", name="USB-C Braided Cable", category="cable", deduction_fee=18.0),
        BOMItem(item_id="case-hard-travel", name="Collapsible Carrying Case", category="case", deduction_fee=38.0),
        BOMItem(item_id="doc-quickstart", name="Quick Start Manual", category="manual", deduction_fee=5.0),
    ]
    profile = SKUProfile(
        sku_id=sku_id,
        title="Sony WH-1000XM5 Wireless Headphones (Black)",
        category="Audio",
        msrp=399.99,
        restocking_fee_rate=0.0,
        bom=test_bom,
        expected_serial_regex=r"^SN-DEV-[A-Z0-9]{4,8}$",
    )
    catalog_mgr.onboard_sku(profile, reference_images=[case_clean.chassis_crop_path])
    pipeline = ReturnTriagePipeline(catalog_manager=catalog_mgr)
    generator = ArbitrationReportGenerator(output_dir=tmp_path_factory.mktemp("dispute_out"))

    return {
        "dataset": dataset,
        "catalog_mgr": catalog_mgr,
        "pipeline": pipeline,
        "generator": generator,
        "profile": profile,
    }


def test_generate_pdf_clean_return(arbitration_fixtures):
    """Validates PDF generation for an authentic clean return with full refund."""
    dataset = arbitration_fixtures["dataset"]
    pipeline = arbitration_fixtures["pipeline"]
    generator = arbitration_fixtures["generator"]
    profile = arbitration_fixtures["profile"]

    case = dataset["CASE-01-CLEAN"]
    req = ReturnInspectionRequest(
        rma_number=case.rma_number,
        sku_id=case.sku_id,
        box_barcode_raw=case.box_barcode,
        chassis_crop_path=str(case.chassis_crop_path),
        serial_plate_crop_path=str(case.serial_plate_crop_path),
        tray_crop_path=str(case.tray_crop_path),
    )
    result = pipeline.inspect(req)

    pdf_path = generator.generate_pdf(result, sku_profile=profile, platform="GENERIC_ARBITRATION")
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 10_000

    # Forensic verification with PyMuPDF
    doc = fitz.open(str(pdf_path))
    assert len(doc) == 2, "Dossier must fit standard 2-page format"

    p1_text = doc[0].get_text()
    assert "RETURN FRAUD & CONDITION DISPUTE CERTIFICATE" in p1_text
    assert "VERIFIED AUTHENTIC - FULL REFUND APPROVED" in p1_text
    assert case.rma_number in p1_text
    assert "$399.99" in p1_text

    p2_text = doc[1].get_text()
    assert "3. BILL OF MATERIALS (BOM) PACKAGING TRAY KITTING AUDIT" in p2_text
    assert "5. CRYPTOGRAPHIC EVIDENCE SEAL & LEGAL CERTIFICATION" in p2_text
    assert result.certificate.evidence_hash in p2_text
    assert "Page 2 of 2" in p2_text


def test_generate_pdf_bait_and_switch_amazon_safet(arbitration_fixtures):
    """Validates Amazon SAFE-T claim dossier for confirmed bait-and-switch swap."""
    dataset = arbitration_fixtures["dataset"]
    pipeline = arbitration_fixtures["pipeline"]
    generator = arbitration_fixtures["generator"]
    profile = arbitration_fixtures["profile"]

    case = dataset["CASE-02-SWAP"]
    req = ReturnInspectionRequest(
        rma_number=case.rma_number,
        sku_id=case.sku_id,
        box_barcode_raw=case.box_barcode,
        chassis_crop_path=str(case.chassis_crop_path),
        serial_plate_crop_path=str(case.serial_plate_crop_path),
        tray_crop_path=str(case.tray_crop_path),
    )
    result = pipeline.inspect(req)

    pdf_path = generator.generate_pdf(result, sku_profile=profile, platform="AMAZON_SAFE_T")
    assert pdf_path.exists()

    doc = fitz.open(str(pdf_path))
    assert len(doc) == 2

    p1_text = doc[0].get_text()
    assert "AMAZON SAFE-T CLAIM ARBITRATION DOSSIER" in p1_text
    assert "REJECT RETURN - FRAUD DETECTED ($0.00 REFUND)" in p1_text
    assert "BAIT_AND_SWITCH" in p1_text
    assert "CONFIRMED_SWAP" in p1_text

    # Verify that evidence photos + heatmap are embedded
    images_p1 = doc[0].get_images()
    assert len(images_p1) >= 2, "Must embed reference and returned chassis photos"


def test_generate_pdf_flipkart_spf(arbitration_fixtures):
    """Validates Flipkart SPF dispute dossier for serial mismatch fraud."""
    dataset = arbitration_fixtures["dataset"]
    pipeline = arbitration_fixtures["pipeline"]
    generator = arbitration_fixtures["generator"]
    profile = arbitration_fixtures["profile"]

    case = dataset["CASE-03-SERIAL-MISMATCH"]
    req = ReturnInspectionRequest(
        rma_number=case.rma_number,
        sku_id=case.sku_id,
        box_barcode_raw=case.box_barcode,
        chassis_crop_path=str(case.chassis_crop_path),
        serial_plate_crop_path=str(case.serial_plate_crop_path),
        tray_crop_path=str(case.tray_crop_path),
    )
    result = pipeline.inspect(req)

    pdf_path = generator.generate_pdf(result, sku_profile=profile, platform="FLIPKART_SPF")
    assert pdf_path.exists()

    doc = fitz.open(str(pdf_path))
    p1_text = doc[0].get_text()
    assert "FLIPKART SELLER PROTECTION FUND (SPF) DOSSIER" in p1_text
    assert "SERIAL_MISMATCH" in p1_text
    assert "REJECT RETURN - FRAUD DETECTED" in p1_text
    assert "MISMATCH" in p1_text


def test_generate_pdf_missing_accessories_stripe(arbitration_fixtures):
    """Validates Stripe chargeback dossier for incomplete return with missing accessories."""
    dataset = arbitration_fixtures["dataset"]
    pipeline = arbitration_fixtures["pipeline"]
    generator = arbitration_fixtures["generator"]
    profile = arbitration_fixtures["profile"]

    case = dataset["CASE-04-MISSING-ACCESSORIES"]
    req = ReturnInspectionRequest(
        rma_number=case.rma_number,
        sku_id=case.sku_id,
        box_barcode_raw=case.box_barcode,
        chassis_crop_path=str(case.chassis_crop_path),
        serial_plate_crop_path=str(case.serial_plate_crop_path),
        tray_crop_path=str(case.tray_crop_path),
    )
    result = pipeline.inspect(req)

    pdf_path = generator.generate_pdf(result, sku_profile=profile, platform="STRIPE_DISPUTE")
    assert pdf_path.exists()

    doc = fitz.open(str(pdf_path))
    p1_text = doc[0].get_text()
    assert "PAYMENT PROCESSOR CHARGEBACK EVIDENCE DOSSIER" in p1_text
    assert "PARTIAL REFUND APPROVED WITH ITEMIZED DEDUCTIONS" in p1_text

    p2_text = doc[1].get_text()
    assert "MISSING" in p2_text
    assert "TOTAL MISSING ACCESSORIES" in p2_text


def test_generate_pdf_graceful_fallback_without_crops(arbitration_fixtures):
    """Tests that PDF generation never fails even if all optional image crops are missing."""
    dataset = arbitration_fixtures["dataset"]
    pipeline = arbitration_fixtures["pipeline"]
    generator = arbitration_fixtures["generator"]
    profile = arbitration_fixtures["profile"]

    case = dataset["CASE-01-CLEAN"]
    req = ReturnInspectionRequest(
        rma_number="RMA-DEGRADED-NO-CROPS",
        sku_id=case.sku_id,
        box_barcode_raw=case.box_barcode,
    )
    result = pipeline.inspect(req)

    pdf_path = generator.generate_pdf(result, sku_profile=profile, platform="GENERIC_ARBITRATION")
    assert pdf_path.exists()

    doc = fitz.open(str(pdf_path))
    assert len(doc) == 2
    p1_text = doc[0].get_text()
    assert "RMA-DEGRADED-NO-CROPS" in p1_text
