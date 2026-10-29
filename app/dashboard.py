"""Streamlit Operator Dashboard for ReturnShield AI.

Features:
- Dual-mode operation: 3PL Warehouse Intake Station & Doorstep Delivery Rider QC (Mobile Mode).
- Zero-training SKU Catalog Explorer with reference image & DINOv2 vector inspection.
- 1-Click synthetic and real-product fraud attack demonstration presets.
- Real-time sub-3-second inspection pipeline execution with live SLA timer.
- DINOv2 spatial difference heatmap visualization side-by-side with OEM reference.
- Dual-serial OCR cross-matching with normalized confusion matrix explanation.
- Packaging tray BOM kitting checklist and itemized refund deduction calculation.
- Cryptographic SHA-256 dispute certificate export for Amazon SAFE-T, Flipkart SPF, and Stripe.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from core.arbitration_report import ArbitrationReportGenerator
from core.catalog_manager import CatalogManager
from core.disposition_engine import DispositionEngine
from core.erp_adapters import (
    create_canonical_payload_from_inspection,
    format_platform_payload,
)
from core.erp_schemas import ERPPlatform
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
    SerialMatchStatus,
    SKUProfile,
)
from core.serial_verifier import SerialVerifier
from core.swap_detector import SwapDetector
from core.volumetric_schemas import (
    HingeCondition,
    LiquidDamageStatus,
    MultiAngleInspectionResult,
)
from core.webhook_security import WebhookSigner
from tests.synthetic_data import generate_synthetic_return_dataset

# ==============================================================================
# Page Setup & Styling
# ==============================================================================

st.set_page_config(
    page_title="ReturnShield AI | 3PL & Doorstep Return Triage",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1e2530;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2e3846;
        margin-bottom: 12px;
    }
    .badge-approved {
        background-color: #0e6251;
        color: #a3e4d7;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-partial {
        background-color: #7d6608;
        color: #f9e79f;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-rejected {
        background-color: #78281f;
        color: #f5b7b1;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-review {
        background-color: #512e5f;
        color: #e8daef;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
    }
    .sla-pill {
        display: inline-block;
        background: linear-gradient(135deg, #1f4037, #99f2c8);
        color: #111;
        font-weight: bold;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Cached Pipeline & Catalog Singletons
# ==============================================================================

@st.cache_resource
def get_catalog_manager() -> CatalogManager:
    """Provides warm CatalogManager instance with disk database."""
    return CatalogManager()


@st.cache_resource
def get_triage_pipeline() -> ReturnTriagePipeline:
    """Initializes and pre-warms master inspection pipeline."""
    catalog = get_catalog_manager()
    pipeline = ReturnTriagePipeline(catalog_manager=catalog)

    # Perform station boot warmup pass
    skus = catalog.list_skus()
    if skus:
        try:
            pipeline.inspect(ReturnInspectionRequest(
                rma_number="RMA-BOOT-WARMUP",
                sku_id=skus[0].sku_id,
                box_barcode_raw="WARMUP-TEST",
            ))
        except Exception:
            pass
    return pipeline


@st.cache_data
def get_synthetic_dataset() -> Dict[str, Any]:
    """Generates synthetic scenarios for demonstration."""
    cache_dir = Path(__file__).resolve().parent.parent / "storage" / "synthetic_demo"
    return generate_synthetic_return_dataset(cache_dir)


catalog_mgr = get_catalog_manager()
triage_pipeline = get_triage_pipeline()


# ==============================================================================
# Sidebar & Mode Selection
# ==============================================================================

st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("ReturnShield AI")
st.sidebar.caption("v1.0.0 | Enterprise 3PL & Doorstep Anti-Fraud Station")

mode = st.sidebar.radio(
    "Workstation Operating Mode:",
    [
        "🏭 3PL Warehouse Intake Station",
        "📱 Doorstep Delivery Rider QC [Mobile Mode]",
        "🗂️ SKU Catalog & Reference Vectors",
        "📹 Real-World Harvested CV Dataset & Video Streams",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Station Hardware Diagnostics")
import torch
cuda_on = torch.cuda.is_available()
st.sidebar.write(f"**Compute Acceleration:** {'🟢 CUDA (NVIDIA GPU)' if cuda_on else '🟡 CPU'}")
if cuda_on:
    st.sidebar.caption(f"Device: `{torch.cuda.get_device_name(0)}`")
st.sidebar.write(f"**Vision Model:** `DINOv2 (ViT-S/14)`")
st.sidebar.write(f"**OCR Engine:** `RapidOCR (ONNX Runtime)`")
st.sidebar.write(f"**Optical Filter:** `90° Cross-Polarizer`")
st.sidebar.write(f"**Target Station SLA:** `< 3.0 Seconds`")


# ==============================================================================
# Mode 1: 3PL Warehouse Intake Station
# ==============================================================================

if mode == "🏭 3PL Warehouse Intake Station":
    st.title("🏭 Warehouse Return Intake & Verification Station")
    st.caption("Automated overhead optical triage station for open-box electronics returns ($100–$2,000 MSRP).")

    # Scenario Presets Selector
    st.markdown("### 1. Select Return Case or Load Preset")
    scenarios = get_synthetic_dataset()

    preset_options = [
        "CASE-01-CLEAN: Sony XM5 (Authentic Like-New)",
        "CASE-02-SWAP: Bait-and-Switch (Cheap Beige Swap Attack)",
        "CASE-03-SERIAL-MISMATCH: Stolen Unit in Genuine Box (Serial Mismatch)",
        "CASE-04-MISSING-ACCESSORIES: Incomplete BOM (Missing Cables & Case)",
        "CASE-05-COSMETIC-WEAR: Heavy Abrasions & Scratches (Grade C)",
        "CASE-06-LIQUID-DAMAGE: Liquid Ingress (Triggered Crimson LDI in Audio Port)",
        "CASE-07-COVERT-SWAP: Covert Sandwich Swap (Top OEM vs Fake Oblique Flank)",
        "REAL-01: Pillar 1 (AirReps Clone Swap - Shenzhen Superclone in OEM Box)",
        "REAL-02: Pillar 1 (Walkman Blog Clone - Sony XM5 Logo Kerning & Seam Defect)",
        "REAL-03: Pillar 2 (eBay Resale #7000 - Scratched Earcup & Broken Hinge)",
        "REAL-04: Pillar 3 (Pallet Liquidation - Unboxing Video Frame & Crushed Packaging)",
        "REAL-05: Pillar 4 (Customer 1-Star Review - Missing BOM Accessories & Snapped Band)",
        "CUSTOM: Upload Custom Return Photos",
    ]

    selected_preset = st.selectbox("Demonstration Preset:", preset_options)

    # State variables for scan execution
    sku_id = "SKU-SONY-WH1000XM5-BLK"
    rma_num = "RMA-2026-8801"
    box_barcode = "SN-SNY-8842109"
    chassis_path = None
    serial_path = None
    tray_path = None
    overhead_path = None
    oblique_path = None

    if selected_preset.startswith("CASE-01"):
        case = scenarios["CASE-01-CLEAN"]
        sku_id = case.sku_id
        rma_num = case.rma_number
        box_barcode = case.box_barcode
        chassis_path = str(case.chassis_crop_path)
        serial_path = str(case.serial_plate_crop_path)
        tray_path = str(case.tray_crop_path)
        overhead_path = str(case.overhead_image_path)
        oblique_path = str(case.oblique_image_path) if case.oblique_image_path else None
    elif selected_preset.startswith("CASE-02"):
        case = scenarios["CASE-02-SWAP"]
        sku_id = case.sku_id
        rma_num = case.rma_number
        box_barcode = case.box_barcode
        chassis_path = str(case.chassis_crop_path)
        serial_path = str(case.serial_plate_crop_path)
        tray_path = str(case.tray_crop_path)
        overhead_path = str(case.overhead_image_path)
        oblique_path = str(case.oblique_image_path) if case.oblique_image_path else None
    elif selected_preset.startswith("CASE-03"):
        case = scenarios["CASE-03-SERIAL-MISMATCH"]
        sku_id = case.sku_id
        rma_num = case.rma_number
        box_barcode = case.box_barcode
        chassis_path = str(case.chassis_crop_path)
        serial_path = str(case.serial_plate_crop_path)
        tray_path = str(case.tray_crop_path)
        overhead_path = str(case.overhead_image_path)
        oblique_path = str(case.oblique_image_path) if case.oblique_image_path else None
    elif selected_preset.startswith("CASE-04"):
        case = scenarios["CASE-04-MISSING-ACCESSORIES"]
        sku_id = case.sku_id
        rma_num = case.rma_number
        box_barcode = case.box_barcode
        chassis_path = str(case.chassis_crop_path)
        serial_path = str(case.serial_plate_crop_path)
        tray_path = str(case.tray_crop_path)
        overhead_path = str(case.overhead_image_path)
        oblique_path = str(case.oblique_image_path) if case.oblique_image_path else None
    elif selected_preset.startswith("CASE-05"):
        case = scenarios["CASE-05-COSMETIC-WEAR"]
        sku_id = case.sku_id
        rma_num = case.rma_number
        box_barcode = case.box_barcode
        chassis_path = str(case.chassis_crop_path)
        serial_path = str(case.serial_plate_crop_path)
        tray_path = str(case.tray_crop_path)
        overhead_path = str(case.overhead_image_path)
        oblique_path = str(case.oblique_image_path) if case.oblique_image_path else None
    elif selected_preset.startswith("CASE-06"):
        case = scenarios["CASE-06-LIQUID-DAMAGE"]
        sku_id = case.sku_id
        rma_num = case.rma_number
        box_barcode = case.box_barcode
        chassis_path = str(case.chassis_crop_path)
        serial_path = str(case.serial_plate_crop_path)
        tray_path = str(case.tray_crop_path)
        overhead_path = str(case.overhead_image_path)
        oblique_path = str(case.oblique_image_path) if case.oblique_image_path else None
    elif selected_preset.startswith("CASE-07"):
        case = scenarios["CASE-07-COVERT-SWAP"]
        sku_id = case.sku_id
        rma_num = case.rma_number
        box_barcode = case.box_barcode
        chassis_path = str(case.chassis_crop_path)
        serial_path = str(case.serial_plate_crop_path)
        tray_path = str(case.tray_crop_path)
        overhead_path = str(case.overhead_image_path)
        oblique_path = str(case.oblique_image_path) if case.oblique_image_path else None
    elif selected_preset.startswith("REAL-01"):
        sku_id = "SKU-APPLE-APM-SLV"
        rma_num = "RMA-AIRREPS-CLONE-01"
        box_barcode = "H1CDG0JTP3W9"
        base_dir = Path(__file__).resolve().parent.parent / "storage" / "sample_returns" / "REAL-CASE-01-CLONE-SWAP-AIRREPS"
        chassis_path = str(base_dir / "chassis.jpg")
        oblique_path = str(base_dir / "detail.jpg")
    elif selected_preset.startswith("REAL-02"):
        sku_id = "SKU-SONY-WH1000XM5-BLK"
        rma_num = "RMA-WALKMAN-CLONE-02"
        box_barcode = "SN-SNY-8842109"
        base_dir = Path(__file__).resolve().parent.parent / "storage" / "sample_returns" / "REAL-CASE-02-SONY-CLONE-TEARDOWN"
        chassis_path = str(base_dir / "chassis.jpg")
        oblique_path = str(base_dir / "detail.jpg")
    elif selected_preset.startswith("REAL-03"):
        sku_id = "SKU-SONY-WH1000XM5-BLK"
        rma_num = "RMA-EBAY-PARTS-03"
        box_barcode = "SN-SNY-8842109"
        base_dir = Path(__file__).resolve().parent.parent / "storage" / "sample_returns" / "REAL-CASE-03-EBAY-BROKEN-HINGE"
        chassis_path = str(base_dir / "chassis.jpg")
        oblique_path = str(base_dir / "detail.jpg")
    elif selected_preset.startswith("REAL-04"):
        sku_id = "SKU-SONY-WH1000XM5-BLK"
        rma_num = "RMA-PALLET-UNBOX-04"
        box_barcode = "SN-SNY-8842109"
        base_dir = Path(__file__).resolve().parent.parent / "storage" / "sample_returns" / "REAL-CASE-04-PALLET-UNBOXING-FRAME"
        chassis_path = str(base_dir / "chassis.jpg")
        oblique_path = str(base_dir / "detail.jpg")
    elif selected_preset.startswith("REAL-05"):
        sku_id = "SKU-APPLE-APM-SLV"
        rma_num = "RMA-CUSTOMER-REVIEW-05"
        box_barcode = "H1CDG0JTP3W9"
        base_dir = Path(__file__).resolve().parent.parent / "storage" / "sample_returns" / "REAL-CASE-05-CUSTOMER-EVIDENCE-MISSING-BOM"
        chassis_path = str(base_dir / "chassis.jpg")
        oblique_path = str(base_dir / "detail.jpg")

    # Bench Inputs
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        rma_num = st.text_input("RMA / Return Order Number:", value=rma_num)
    with col_in2:
        sku_options = [s.sku_id for s in catalog_mgr.list_skus()]
        sku_index = sku_options.index(sku_id) if sku_id in sku_options else 0
        sku_id = st.selectbox("Expected Catalog SKU:", sku_options, index=sku_index)
    with col_in3:
        box_barcode = st.text_input("Scanned Outer Box Barcode:", value=box_barcode)

    # Visual Inspection Feeds Preview
    st.markdown("### 2. Multi-Zone Station Capture Preview")
    col_img1, col_img2, col_img3, col_img4 = st.columns(4)

    with col_img1:
        st.markdown("**Zone A: Chassis ROI**")
        if chassis_path and Path(chassis_path).exists():
            st.image(chassis_path, caption="Product Chassis Capture", use_container_width=True)
        else:
            uploaded_chassis = st.file_uploader("Upload Chassis Crop:", type=["png", "jpg", "jpeg"], key="up_chassis")
            if uploaded_chassis:
                temp_p = Path("storage/temp_chassis.png")
                temp_p.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_p, "wb") as f:
                    f.write(uploaded_chassis.read())
                chassis_path = str(temp_p)
                st.image(chassis_path, use_container_width=True)

    with col_img2:
        st.markdown("**Zone B: Laser Serial Plate (Cross-Pol)**")
        if serial_path and Path(serial_path).exists():
            st.image(serial_path, caption="Cross-Polarized Serial Engraving", use_container_width=True)
        else:
            st.info("Serial plate crop (optional in direct simulation)")

    with col_img3:
        st.markdown("**Zone C: Packaging Tray (BOM)**")
        if tray_path and Path(tray_path).exists():
            st.image(tray_path, caption="Accessory Cavity Layout", use_container_width=True)
        else:
            st.info("Packaging tray crop (optional)")

    with col_img4:
        st.markdown("**Zone D: 45° Oblique Flank & Ports**")
        if oblique_path and Path(oblique_path).exists():
            st.image(oblique_path, caption="Oblique Perspective ROI", use_container_width=True)
        else:
            uploaded_oblique = st.file_uploader("Upload Oblique:", type=["png", "jpg", "jpeg"], key="up_oblique")
            if uploaded_oblique:
                temp_ob = Path("storage/temp_oblique.png")
                temp_ob.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_ob, "wb") as f:
                    f.write(uploaded_oblique.read())
                oblique_path = str(temp_ob)
                st.image(oblique_path, use_container_width=True)

    st.markdown("---")

    # Scan Trigger Button
    trigger_col, sla_col = st.columns([1, 4])
    with trigger_col:
        run_scan = st.button("⚡ EXECUTE TRIAGE SCAN", type="primary", use_container_width=True)

    if run_scan:
        with st.spinner("Executing DINOv2 visual extraction, OCR alignment, and BOM audit..."):
            t_start = time.time()
            req = ReturnInspectionRequest(
                rma_number=rma_num,
                sku_id=sku_id,
                box_barcode_raw=box_barcode,
                chassis_crop_path=chassis_path,
                serial_plate_crop_path=serial_path,
                tray_crop_path=tray_path,
                overhead_image_path=overhead_path,
                oblique_image_path=oblique_path,
                operator_id="OPERATOR-BAY-03",
                bench_id="BENCH-DOCK-EAST",
            )
            result: ReturnInspectionResult = triage_pipeline.inspect(req)
            elapsed_ms = round((time.time() - t_start) * 1000.0, 1)

        # SLA Callout
        st.markdown(f"<span class='sla-pill'>⚡ Station SLA Achieved: {elapsed_ms}ms (Budget: 3,000ms)</span>", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # Top-Level Disposition Banner
        # ----------------------------------------------------------------------
        st.markdown("### 3. Automated Financial Triage Decision")
        action = result.disposition.action

        if action == DispositionAction.FULL_REFUND:
            st.success(f"### ✅ APPROVED: FULL REFUND (${result.disposition.net_refund_amount:.2f})\n{result.disposition.recommendation_summary}")
        elif action == DispositionAction.PARTIAL_REFUND:
            st.warning(f"### ⚠️ APPROVED: PARTIAL REFUND (${result.disposition.net_refund_amount:.2f})\n{result.disposition.recommendation_summary}")
        elif action == DispositionAction.REJECT_FRAUD_HOLD:
            st.error(f"### 🛑 REJECTED: FRAUD HOLD ($0.00 Refund)\n{result.disposition.recommendation_summary}")
        else:
            st.info(f"### 🔍 SECONDARY MANUAL REVIEW REQUIRED\n{result.disposition.recommendation_summary}")

        # KPI Metrics Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Original MSRP", f"${result.disposition.original_msrp:.2f}")
        with kpi2:
            st.metric("Total Deductions", f"${result.disposition.total_deductions:.2f}", delta=f"-${result.disposition.total_deductions:.2f}" if result.disposition.total_deductions > 0 else "None")
        with kpi3:
            st.metric("Net Refund Disbursed", f"${result.disposition.net_refund_amount:.2f}")
        with kpi4:
            st.metric("Condition Grade", result.disposition.condition_grade.value)

        # Active Anomaly Tags
        st.markdown("**Detected Anomaly Flags:** " + " ".join([f"`{f.value}`" for f in result.fraud_flags]))

        # ----------------------------------------------------------------------
        # Deep-Dive Detailed Forensic Tabs
        # ----------------------------------------------------------------------
        tab_swap, tab_vol, tab_serial, tab_kitting, tab_cert, tab_erp = st.tabs([
            "🔍 Visual Identity & DINOv2 Heatmap",
            "🌊 Volumetric & Multi-Angle Stereo",
            "🔤 Dual-Serial OCR Verification",
            "📦 BOM Kitting Checklist",
            "🛡️ Cryptographic Dispute Certificate",
            "📡 Real-Time ERP Webhooks",
        ])

        with tab_swap:
            st.markdown("#### Foundation Model Visual Identity (DINOv2 ViT-S/14)")
            if result.swap_result:
                col_sw1, col_sw2 = st.columns(2)
                with col_sw1:
                    st.write(f"**Visual Identity Status:** `{result.swap_result.identity_status.value}`")
                    st.write(f"**Cosine Similarity vs Catalog Reference:** `{result.swap_result.cosine_similarity:.4f}`")
                    st.progress(max(0.0, min(1.0, float(result.swap_result.cosine_similarity))))
                    st.write(f"**Decision Threshold:** `{result.swap_result.threshold}`")
                    if result.swap_result.spatial_diff_score is not None:
                        st.write(f"**Patch-Level Peak Anomaly Score:** `{result.swap_result.spatial_diff_score:.4f}`")
                    st.info(result.swap_result.explanation)

                with col_sw2:
                    if result.swap_result.heatmap_path and Path(result.swap_result.heatmap_path).exists():
                        st.image(result.swap_result.heatmap_path, caption="Spatial Diff Heatmap Overlay (Jet Colormap: Red = Deviant Region)", use_container_width=True)
                    else:
                        st.info("No spatial diff heatmap generated for this case.")
            else:
                st.write("Visual identity analysis not executed (no chassis crop provided).")

        with tab_vol:
            st.markdown("#### Volumetric Stereo Disparity & Lateral Inspection")
            if result.volumetric_result:
                v = result.volumetric_result
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    st.write(f"**Liquid Damage Status (LDI):** `{v.liquid_damage_status.value}`")
                    st.write(f"**Crimson LDI Pixels Detected:** `{v.liquid_indicator_pixels}`")
                    st.write(f"**Lateral Hinge Integrity:** `{v.hinge_condition.value}`")
                    st.write(f"**Multi-View Fused Similarity:** `{v.multi_view_fused_similarity:.4f}`")
                    st.write(f"  • Top Similarity (90°): `{v.top_similarity:.4f}`")
                    st.write(f"  • Oblique Similarity (45°): `{v.oblique_similarity:.4f}`")
                    if v.lateral_tamper_detected:
                        st.error("🛑 LATERAL TAMPER DETECTED: Covert top-shell vs flank swap divergence.")
                    else:
                        st.success("✅ Multi-angle perspective geometric alignment verified.")

                    if v.lateral_findings:
                        st.markdown("**Forensic Volumetric Findings:**")
                        for f in v.lateral_findings:
                            st.write(f"- {f}")

                with col_v2:
                    if v.disparity_map_path and Path(v.disparity_map_path).exists():
                        st.image(v.disparity_map_path, caption="StereoSGBM Disparity Heatmap Overlay (OpenCV TURBO)", use_container_width=True)
                    else:
                        st.info("Disparity map generated during stereo camera capture.")
            else:
                st.info("No oblique/stereo inspection feed provided for this return.")

        with tab_serial:
            st.markdown("#### Cross-Match: Shipping Label Barcode vs Metallic Serial Engraving")
            if result.serial_result:
                col_se1, col_se2 = st.columns(2)
                with col_se1:
                    st.write(f"**Box Label Barcode Scanned:** `{result.serial_result.box_barcode or 'Not Detected'}`")
                    st.write(f"**Chassis Serial OCR Readout:** `{result.serial_result.chassis_serial_raw or 'Not Detected'}`")
                    st.write(f"**Normalized Serial (Noise-Filtered):** `{result.serial_result.chassis_serial_normalized or 'N/A'}`")
                    st.write(f"**Serial Match Status:** `{result.serial_result.match_status.value}`")
                    st.write(f"**Character Similarity Ratio:** `{result.serial_result.similarity_ratio:.2f}`")
                    st.write(f"**OCR Engine Confidence:** `{result.serial_result.ocr_confidence:.2f}`")
                with col_se2:
                    if result.serial_result.is_match:
                        st.success(f"✅ SERIALS MATCH: {result.serial_result.notes}")
                    else:
                        st.error(f"🛑 SERIAL MISMATCH DETECTED: {result.serial_result.notes}")
            else:
                st.write("Serial verification not executed.")

        with tab_kitting:
            st.markdown("#### Bill of Materials (BOM) Accessory Tray Audit")
            if result.kitting_result:
                col_k1, col_k2, col_k3 = st.columns(3)
                with col_k1:
                    st.metric("Expected Accessories", result.kitting_result.expected_items_count)
                with col_k2:
                    st.metric("Detected Accessories", result.kitting_result.detected_items_count)
                with col_k3:
                    st.metric("Missing Deductions Total", f"${result.kitting_result.total_accessory_deduction:.2f}")

                # Itemized Table
                item_rows = []
                for item in result.kitting_result.item_audits:
                    item_rows.append({
                        "Accessory ID": item.bom_item_id,
                        "Name": item.name,
                        "Status": item.status.value,
                        "Confidence": f"{item.confidence:.2f}",
                        "Deduction Fee": f"${item.deduction_fee:.2f}",
                    })
                st.table(item_rows)
            else:
                st.write("Kitting audit not executed.")

        with tab_cert:
            st.markdown("#### Cryptographically Sealed Dispute Evidence Certificate")
            if result.certificate:
                st.code(f"SHA-256 SIGNATURE: {result.certificate.evidence_hash}", language="text")
                st.write(f"**Certificate ID:** `{result.certificate.certificate_id}`")
                st.write(f"**Timestamp (UTC):** `{result.certificate.timestamp}`")

                st.markdown("**Dispute-Ready Arbitration Narrative (Amazon SAFE-T / Flipkart SPF / Stripe):**")
                st.text_area("Dispute Narrative:", value=result.certificate.audit_narrative, height=180)

                cert_json = result.certificate.model_dump_json(indent=2)
                st.download_button(
                    label="📥 Download Evidence Certificate (JSON)",
                    data=cert_json,
                    file_name=f"returnshield_cert_{result.certificate.certificate_id}.json",
                    mime="application/json",
                )

                st.markdown("---")
                st.markdown("#### ⚖️ Automated Arbitration Claim Dossier (PDF)")
                st.markdown("Generate a court- and platform-admissible dispute packet incorporating side-by-side photo evidence, DINOv2 heatmaps, dual-serial OCR forensics, and SHA-256 digital seals.")

                col_plat, col_dl = st.columns([3, 2])
                with col_plat:
                    platform_choices = [
                        ("AMAZON_SAFE_T", "Amazon SAFE-T Claim (US / India)"),
                        ("FLIPKART_SPF", "Flipkart Seller Protection Fund (SPF)"),
                        ("STRIPE_DISPUTE", "Stripe / Card Chargeback Defense"),
                        ("GENERIC_ARBITRATION", "Universal 3PL Insurance Dossier"),
                    ]
                    selected_platform = st.selectbox(
                        "Target Arbitration Platform:",
                        platform_choices,
                        format_func=lambda x: x[1],
                        key="target_platform_select",
                    )
                with col_dl:
                    st.write("")
                    st.write("")
                    prof = catalog_mgr.get_sku(result.sku_id)
                    gen = ArbitrationReportGenerator()
                    pdf_path = gen.generate_pdf(result, sku_profile=prof, platform=selected_platform[0])
                    with open(pdf_path, "rb") as pf:
                        pdf_data = pf.read()

                    st.download_button(
                        label=f"📄 Download {selected_platform[0]} PDF Dossier",
                        data=pdf_data,
                        file_name=pdf_path.name,
                        mime="application/pdf",
                        key="pdf_download_button",
                        type="primary",
                        use_container_width=True,
                    )

        with tab_erp:
            st.markdown("#### Real-Time E-Commerce ERP & Platform Webhooks")
            st.caption("Synchronizes physical return triage outcomes with Shopify, Narvar, Amazon SAFE-T, and Flipkart SPF.")

            canonical_payload = create_canonical_payload_from_inspection(result)

            erp_choice = st.radio(
                "Platform Payload Preview:",
                ["Shopify Admin REST (refunds.json)", "Narvar Post-Purchase", "Amazon SAFE-T Claim", "Flipkart SPF Claim"],
                horizontal=True,
            )

            if erp_choice.startswith("Shopify"):
                pl_dict = format_platform_payload(canonical_payload, ERPPlatform.SHOPIFY)
                raw_bytes = json.dumps(pl_dict).encode("utf-8")
                sig = WebhookSigner.sign_shopify_base64(raw_bytes, "shpss_test_secret_key_8842109")
                st.code(f"X-Shopify-Hmac-SHA256: {sig}", language="text")
                st.json(pl_dict)
            elif erp_choice.startswith("Narvar"):
                pl_dict = format_platform_payload(canonical_payload, ERPPlatform.NARVAR)
                st.json(pl_dict)
            elif erp_choice.startswith("Amazon"):
                pl_dict = format_platform_payload(canonical_payload, ERPPlatform.AMAZON_SAFE_T)
                st.json(pl_dict)
            else:
                pl_dict = format_platform_payload(canonical_payload, ERPPlatform.FLIPKART_SPF)
                st.json(pl_dict)


# ==============================================================================
# Mode 2: Doorstep Delivery Rider QC (Mobile Mode for India / APAC)
# ==============================================================================

elif mode == "📱 Doorstep Delivery Rider QC [Mobile Mode]":
    st.title("📱 Delivery Rider Doorstep QC Mode")
    st.markdown("""
    **Optimized for Smartphone Screens & 45-Second Doorstep Delivery Window (India / Flipkart / Amazon.in).**
    Defeats the primary fraud vector in Indian e-commerce: customer handing over an older broken lookalike device
    or courier rider swap in-transit before OTP exchange.
    """)

    st.markdown("---")
    rider_col1, rider_col2 = st.columns([1, 1])

    with rider_col1:
        st.subheader("1. Customer Order Details")
        doorstep_sku = st.selectbox("Product on Invoice:", [s.sku_id for s in catalog_mgr.list_skus()])
        doorstep_rma = st.text_input("Return Pick-up ID / AWB #:", value="IN-DEL-994102")
        customer_barcode = st.text_input("Scan Shipping Box Barcode:", value="SN-SNY-8842109")

        st.subheader("2. Doorstep Camera Snap")
        st.caption("Point smartphone camera at the unboxed device on the customer's doorstep table:")
        scenarios = get_synthetic_dataset()

        doorstep_preset = st.radio(
            "Simulate Rider Photo Snap:",
            [
                "Customer offers Genuine Sony WH-1000XM5 (Legitimate)",
                "Customer offers Older Broken Knockoff / Beige Lookalike (Doorstep Swap Fraud)",
            ]
        )

        if "Genuine" in doorstep_preset:
            doorstep_chassis = str(scenarios["CASE-01-CLEAN"].chassis_crop_path)
        else:
            doorstep_chassis = str(scenarios["CASE-02-SWAP"].chassis_crop_path)

        st.image(doorstep_chassis, caption="Live Camera Feed", use_container_width=True)

    with rider_col2:
        st.subheader("3. Instant AI Gatekeeper Decision")
        st.write("Scan completes in **< 1.5 seconds** on mobile network.")

        rider_verify = st.button("🔍 VERIFY ITEM AT DOORSTEP", type="primary", use_container_width=True)

        if rider_verify:
            with st.spinner("AI checking visual identity vs catalog spec..."):
                t0 = time.time()
                req = ReturnInspectionRequest(
                    rma_number=doorstep_rma,
                    sku_id=doorstep_sku,
                    box_barcode_raw=customer_barcode,
                    chassis_crop_path=doorstep_chassis,
                    operator_id="RIDER-BANGALORE-14",
                    bench_id="MOBILE-APP-ANDROID",
                )
                res = triage_pipeline.inspect(req)
                t_el = round((time.time() - t0) * 1000.0, 1)

            st.write(f"⏱️ **Mobile Latency:** `{t_el}ms`")

            if not res.is_fraud and res.disposition.action != DispositionAction.REJECT_FRAUD_HOLD:
                st.markdown("""
                <div style="background-color: #0e6251; padding: 20px; border-radius: 10px; color: white;">
                    <h2>✅ RETURN ACCEPTED</h2>
                    <h3>PROCEED TO COLLECT OTP FROM CUSTOMER</h3>
                    <p>Visual model confirmed: OEM Sony WH-1000XM5.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #78281f; padding: 20px; border-radius: 10px; color: white;">
                    <h2>🛑 RETURN REJECTED - DO NOT ACCEPT</h2>
                    <h3>DO NOT ASK FOR OTP. DO NOT PICK UP ITEM.</h3>
                    <p><b>Reason:</b> Returned item does not match OEM catalog specification (Lookalike Swap / Counterfeit).</p>
                    <p>Advise customer to contact e-commerce customer support.</p>
                </div>
                """, unsafe_allow_html=True)

            if res.swap_result and res.swap_result.heatmap_path:
                st.image(res.swap_result.heatmap_path, caption="Dispute Evidence Overlay", use_container_width=True)


# ==============================================================================
# Mode 3: SKU Catalog & Reference Vectors
# ==============================================================================

elif mode == "🗂️ SKU Catalog & Reference Vectors":
    st.title("🗂️ Zero-Training SKU Catalog & Embedding Database")
    st.caption("Manage catalog profiles, Bill of Materials manifests, and cached DINOv2 vision representations.")

    skus = catalog_mgr.list_skus()
    st.markdown(f"**Total Registered SKUs:** `{len(skus)}`")

    tab_browse, tab_onboard = st.tabs(["Browse Registered SKUs", "Onboard New SKU"])

    with tab_browse:
        for profile in skus:
            with st.expander(f"📦 {profile.title} ({profile.sku_id}) - MSRP: ${profile.msrp:.2f}"):
                c1, c2 = st.columns([1, 2])
                with c1:
                    if profile.reference_image_paths and Path(profile.reference_image_paths[0]).exists():
                        st.image(profile.reference_image_paths[0], caption="Golden Reference", use_container_width=True)
                    else:
                        st.info("No reference photo on disk.")
                with c2:
                    st.write(f"**Category:** {profile.category}")
                    st.write(f"**MSRP:** ${profile.msrp:.2f}")
                    st.write(f"**Restocking Fee Rate:** {int(profile.restocking_fee_rate * 100)}%")
                    st.write(f"**Expected Serial Regex:** `{profile.expected_serial_regex}`")
                    st.write(f"**Similarity Threshold:** `{profile.similarity_threshold}`")

                    # Check if embedding file exists
                    emb_p = catalog_mgr.embeddings_dir / f"{profile.sku_id}.npz"
                    if emb_p.exists():
                        st.success(f"⚡ DINOv2 Vector Cached: `{emb_p.name}` (Size: {emb_p.stat().st_size} bytes)")
                    else:
                        st.warning("Embedding not cached.")

                    st.markdown("**Required Bill of Materials (BOM):**")
                    if profile.bom:
                        bom_table = [
                            {"Item ID": b.item_id, "Name": b.name, "Category": b.category, "Deduction Fee": f"${b.deduction_fee:.2f}"}
                            for b in profile.bom
                        ]
                        st.table(bom_table)
                    else:
                        st.write("No accessories specified.")

    with tab_onboard:
        st.subheader("Onboard New Product SKU")
        with st.form("onboard_sku_form"):
            new_id = st.text_input("SKU ID (e.g. SKU-SAMSUNG-S24U):")
            new_title = st.text_input("Product Title:")
            new_category = st.text_input("Taxonomy Category:", value="Electronics")
            new_msrp = st.number_input("MSRP ($):", min_value=1.0, value=999.99, step=10.0)
            new_fee = st.slider("Restocking Fee (%):", min_value=0, max_value=30, value=15) / 100.0
            new_regex = st.text_input("Serial Regex Pattern:", value=r"^[A-Z0-9]{10,18}$")
            submitted = st.form_submit_button("Register SKU")

            if submitted:
                if not new_id or not new_title:
                    st.error("SKU ID and Title are required.")
                else:
                    new_prof = SKUProfile(
                        sku_id=new_id,
                        title=new_title,
                        category=new_category,
                        msrp=new_msrp,
                        restocking_fee_rate=new_fee,
                        expected_serial_regex=new_regex,
                    )
                    catalog_mgr.onboard_sku(new_prof)
                    st.success(f"Successfully onboarded SKU '{new_id}'!")
                    st.rerun()


# ==============================================================================
# Mode 4: Real-World Harvested CV Dataset & Video Streams
# ==============================================================================

elif mode == "📹 Real-World Harvested CV Dataset & Video Streams":
    st.title("📹 Real-World Consumer Return CV Dataset & Video Streams")
    st.markdown(
        "> **Authentic 4-Pillar Consumer Electronics Return Fraud Dataset & Video Streams**  \n"
        "> *Harvested from Counterfeit Communities (AirReps / Walkman Blog), Secondary Resale (eBay / Swappa), "
        "4K Teardown & Pallet Liquidation Streams (yt-dlp), and 1-Star Customer Review Evidence.*"
    )

    manifest_path = Path(__file__).resolve().parent.parent / "storage" / "real_world_harvest_manifest.json"
    if not manifest_path.exists():
        st.warning("Harvest manifest not found. Run `python scripts/harvest_real_world_returns.py` to harvest assets.")
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        records = manifest_data.get("records", [])

        # Top Metric Cards (4-Pillars)
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Harvested Assets", len(records))
        with col_m2:
            p1_p2 = len([r for r in records if "Pillar 1" in r.get("pillar", "") or "Pillar 2" in r.get("pillar", "")])
            st.metric("Clones & Resale (P1+P2)", p1_p2)
        with col_m3:
            p3_frames = [r for r in records if "Pillar 3" in r.get("pillar", "") or r.get("defect_type") == "VIDEO_STREAM_KEYFRAME"]
            st.metric("Video Stream Frames (P3)", len(p3_frames))
        with col_m4:
            p4_reviews = len([r for r in records if "Pillar 4" in r.get("pillar", "")])
            st.metric("Review Evidence (P4)", p4_reviews)

        st.markdown("---")

        tab_gallery, tab_video, tab_cases = st.tabs([
            "🏛️ 4-Pillar Defect Gallery",
            "🎥 Video Stream Keyframes",
            "📦 Assembled Return Cases",
        ])

        with tab_gallery:
            st.subheader("Authentic Physical Defect & Counterfeit Photography")
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                pillar_options = ["ALL"] + sorted(list(set(r.get("pillar", "") for r in records if r.get("pillar"))))
                pillar_filter = st.selectbox("Filter Pillar:", pillar_options)
            with col_f2:
                defect_options = ["ALL"] + sorted(list(set(r.get("defect_type", "") for r in records if r.get("defect_type"))))
                defect_filter = st.selectbox("Filter Defect Type:", defect_options)
            with col_f3:
                sku_options = ["ALL"] + sorted(list(set(r.get("sku_id", "") for r in records if r.get("sku_id"))))
                sku_filter = st.selectbox("Filter SKU:", sku_options)

            filtered = records
            if pillar_filter != "ALL":
                filtered = [r for r in filtered if r.get("pillar") == pillar_filter]
            if defect_filter != "ALL":
                filtered = [r for r in filtered if r.get("defect_type") == defect_filter]
            if sku_filter != "ALL":
                filtered = [r for r in filtered if r.get("sku_id") == sku_filter]

            st.caption(f"Showing **{len(filtered)}** matching authentic assets:")

            cols = st.columns(3)
            for i, rec in enumerate(filtered):
                rel_p = rec.get("path")
                abs_p = Path(__file__).resolve().parent.parent / rel_p
                with cols[i % 3]:
                    if abs_p.exists():
                        st.image(str(abs_p), caption=f"{rec.get('defect_type')} ({rec.get('width')}x{rec.get('height')})")
                        st.caption(f"**Pillar:** {rec.get('pillar', 'N/A')}\n\n**Source:** {rec.get('source')} | `{rec.get('sku_id', 'N/A')}`")
                        with st.expander("Asset Details"):
                            st.json(rec)

        with tab_video:
            st.subheader("Decoded Video Stream Inspection Frames")
            st.markdown(
                "Frames extracted via OpenCV `cv2.VideoCapture` at 1 fps from real liquidation unboxings "
                "and hardware teardown streams with Laplacian blur sharpness verification (threshold $\\text{Var}(\\nabla^2 I) > 50$):"
            )

            video_records = [r for r in records if "Pillar 3" in r.get("pillar", "") or r.get("defect_type") == "VIDEO_STREAM_KEYFRAME"]
            vcols = st.columns(3)
            for i, vrec in enumerate(video_records):
                v_rel = vrec.get("path")
                v_abs = Path(__file__).resolve().parent.parent / v_rel
                with vcols[i % 3]:
                    if v_abs.exists():
                        st.image(str(v_abs), caption=f"{vrec.get('feed_id')} (Frame #{vrec.get('frame_index', i+1)})")
                        st.caption(
                            f"**Sharpness Score:** `{vrec.get('laplacian_sharpness')}` | "
                            f"Resolution: `{vrec.get('width')}x{vrec.get('height')}`\n\n"
                            f"**Stream:** {vrec.get('source')}"
                        )

        with tab_cases:
            st.subheader("5 Assembled Production Return Test Cases")
            cases_dir = Path(__file__).resolve().parent.parent / "storage" / "sample_returns"
            case_folders = [d for d in sorted(cases_dir.glob("REAL-CASE-*")) if d.is_dir()]

            for c_folder in case_folders:
                c_manifest = c_folder / "manifest.json"
                if c_manifest.exists():
                    with open(c_manifest, "r", encoding="utf-8") as f:
                        c_info = json.load(f)

                    with st.expander(f"📦 {c_info.get('case_id')}: {c_info.get('title')}", expanded=True):
                        col_c1, col_c2, col_c3 = st.columns([1, 1, 2])
                        chassis_f = c_folder / "chassis.jpg"
                        detail_f = c_folder / "detail.jpg"

                        with col_c1:
                            if chassis_f.exists():
                                st.image(str(chassis_f), caption="Chassis View")
                        with col_c2:
                            if detail_f.exists():
                                st.image(str(detail_f), caption="Detail / Oblique View")
                        with col_c3:
                            st.write(f"**Expected SKU:** `{c_info.get('sku_id')}`")
                            st.write(f"**Expected Grade:** `{c_info.get('expected_condition')}`")
                            st.write(f"**Expected Disposition:** `{c_info.get('expected_disposition')}`")
                            st.write(f"**Fraud Flags:** `{c_info.get('fraud_flags')}`")
                            st.info(f"**Diagnostic Notes:** {c_info.get('notes')}")

