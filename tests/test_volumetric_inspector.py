"""Unit and Integration Tests for Multi-Angle Volumetric & Stereo Inspection.

Validates:
1. OpenCV StereoSGBM disparity calculation and 3D depth profiling.
2. Dual-band HSV Liquid Damage Indicator (LDI) detection (clean white vs crimson triggered).
3. Lateral hinge fracture detection via Sobel gradient energy.
4. Multi-view feature fusion exposing covert top-housing shell swaps.
5. End-to-end ReturnTriagePipeline integration with oblique camera feeds.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image

from core.catalog_manager import CatalogManager
from core.pipeline import ReturnTriagePipeline
from core.schemas import (
    ConditionGrade,
    DispositionAction,
    FraudFlag,
    ReturnInspectionRequest,
)
from core.volumetric_inspector import VolumetricStereoInspector
from core.volumetric_schemas import (
    HingeCondition,
    LiquidDamageStatus,
    MultiAngleInspectionResult,
)


@pytest.fixture
def inspector(tmp_path: Path) -> VolumetricStereoInspector:
    return VolumetricStereoInspector(heatmaps_dir=tmp_path / "heatmaps")


@pytest.fixture
def stereo_pair_pattern() -> tuple[np.ndarray, np.ndarray]:
    """Generates a synthetic rectified stereo pair with known horizontal disparity."""
    h, w = 240, 320
    left = np.zeros((h, w, 3), dtype=np.uint8) + 40
    right = np.zeros((h, w, 3), dtype=np.uint8) + 40

    # Draw high-contrast textured square in left image at (100, 80)
    # and shifted to (90, 80) in right image (disparity = 10 px)
    cv2.rectangle(left, (100, 80), (180, 160), (220, 220, 220), -1)
    cv2.circle(left, (140, 120), 20, (50, 50, 50), -1)

    cv2.rectangle(right, (90, 80), (170, 160), (220, 220, 220), -1)
    cv2.circle(right, (130, 120), 20, (50, 50, 50), -1)

    return left, right


# ==============================================================================
# 1. StereoSGBM Disparity & Depth Profiling Tests
# ==============================================================================

def test_stereosgbm_disparity_computation(inspector: VolumetricStereoInspector, stereo_pair_pattern):
    """Validates OpenCV StereoSGBM disparity calculation and depth discrepancy calculation."""
    left, right = stereo_pair_pattern

    disp_fp, heatmap_path, profile = inspector.compute_disparity_map(
        left_image=left,
        right_image=right,
        save_heatmap=True,
        nominal_depth_mm=45.0,
        depth_tolerance_mm=10.0,
    )

    assert disp_fp.shape == (240, 320)
    assert heatmap_path is not None
    assert Path(heatmap_path).exists()
    assert profile.measured_depth_mm > 0.0
    assert profile.nominal_depth_mm == 45.0
    assert isinstance(profile.discrepancy_detected, bool)


# ==============================================================================
# 2. Liquid Damage Indicator (LDI) Detection Tests
# ==============================================================================

def test_liquid_damage_indicator_clean(inspector: VolumetricStereoInspector):
    """Pristine white/silver moisture sticker should yield NO_LIQUID_DETECTED."""
    # Create white/silver sticker patch (RGB: 230, 230, 230)
    clean_patch = np.full((60, 60, 3), 230, dtype=np.uint8)
    status, pixels = inspector.inspect_liquid_damage_indicator(clean_patch)

    assert status == LiquidDamageStatus.NO_LIQUID_DETECTED
    assert pixels == 0


def test_liquid_damage_indicator_triggered(inspector: VolumetricStereoInspector):
    """Moisture-triggered crimson/pink indicator sticker should yield CONFIRMED_LIQUID_DAMAGE."""
    # Create patch with bright crimson/red LDI center (BGR: [30, 30, 220] in OpenCV)
    triggered_patch = np.full((60, 60, 3), 200, dtype=np.uint8)
    # Paint central circle with crimson liquid contact hue
    cv2.circle(triggered_patch, (30, 30), 18, (30, 20, 220), -1)

    status, pixels = inspector.inspect_liquid_damage_indicator(
        triggered_patch, red_pixel_trigger_threshold=50
    )

    assert status == LiquidDamageStatus.CONFIRMED_LIQUID_DAMAGE
    assert pixels >= 50


# ==============================================================================
# 3. Lateral Hinge Fracture Analysis Tests
# ==============================================================================

def test_hinge_integrity_clean_vs_fractured(inspector: VolumetricStereoInspector):
    """Validates hairline fracture detection across pivot axis using Sobel energy."""
    # 1. Smooth intact OEM hinge
    clean_hinge = np.zeros((80, 80, 3), dtype=np.uint8) + 128
    cv2.line(clean_hinge, (10, 40), (70, 40), (140, 140, 140), 2)  # subtle smooth seam

    status_clean, energy_clean = inspector.inspect_hinge_integrity(clean_hinge)
    assert status_clean == HingeCondition.INTACT_OEM

    # 2. Fractured hinge with high-gradient jagged crack
    fractured_hinge = np.zeros((80, 80, 3), dtype=np.uint8) + 128
    # Draw intense high-contrast jagged crack line
    pts = np.array([[10, 38], [25, 48], [40, 32], [55, 46], [70, 35]], np.int32)
    cv2.polylines(fractured_hinge, [pts], isClosed=False, color=(255, 255, 255), thickness=4)
    cv2.polylines(fractured_hinge, [pts + 1], isClosed=False, color=(0, 0, 0), thickness=2)

    status_broken, energy_broken = inspector.inspect_hinge_integrity(
        fractured_hinge, fracture_energy_threshold=30.0
    )
    assert status_broken == HingeCondition.FRACTURED_CRACKED
    assert energy_broken > energy_clean


# ==============================================================================
# 4. Multi-View Feature Fusion Tests
# ==============================================================================

def test_multi_view_feature_fusion(inspector: VolumetricStereoInspector):
    """Exposes covert flank swaps when top shell is genuine but oblique view fails."""
    # Authentic product: both top and oblique views match
    fused_clean, tamper_clean = inspector.fuse_multi_view_inspection(
        top_similarity=0.94,
        oblique_similarity=0.91,
    )
    assert fused_clean >= 0.90
    assert tamper_clean is False

    # Covert shell swap: genuine top shell (0.92) but fake oblique flank (0.58)
    fused_swap, tamper_swap = inspector.fuse_multi_view_inspection(
        top_similarity=0.92,
        oblique_similarity=0.58,
        swap_threshold=0.75,
    )
    assert tamper_swap is True
    assert fused_swap < 0.80


# ==============================================================================
# 5. End-to-End Pipeline Multi-Angle Triage Integration
# ==============================================================================

def test_pipeline_with_oblique_inspection_liquid_damage(tmp_path: Path):
    """Verifies that an oblique image with liquid damage triggers deductions in ReturnTriagePipeline."""
    pipeline = ReturnTriagePipeline()

    # Use authentic Sony XM5 reference image as top chassis
    top_ref = Path("storage/catalog_images/SKU-SONY-WH1000XM5-BLK/ref_1.png")
    if top_ref.exists():
        top_img_path = top_ref
    else:
        top_img_path = tmp_path / "top_chassis.png"
        cv2.imwrite(str(top_img_path), np.full((224, 224, 3), 100, dtype=np.uint8))

    # Generate oblique image using authentic reference as base with triggered crimson LDI
    oblique_ref = Path("storage/catalog_images/SKU-SONY-WH1000XM5-BLK/ref_2.png")
    if not oblique_ref.exists():
        oblique_ref = top_ref

    if oblique_ref.exists():
        oblique_arr = cv2.imread(str(oblique_ref))
    else:
        oblique_arr = np.full((224, 224, 3), 120, dtype=np.uint8)

    # Crimson LDI trigger (small patch in port / indicator area)
    oblique_img_path = tmp_path / "oblique_flank.png"
    cv2.circle(oblique_arr, (oblique_arr.shape[1] // 2, oblique_arr.shape[0] // 2), 15, (20, 20, 230), -1)
    cv2.imwrite(str(oblique_img_path), oblique_arr)

    req = ReturnInspectionRequest(
        rma_number="RMA-VOL-001",
        sku_id="SKU-SONY-WH1000XM5-BLK",
        box_barcode_raw="SN-SONY-001928",
        chassis_crop_path=str(top_img_path),
        oblique_image_path=str(oblique_img_path),
    )

    result = pipeline.inspect(req)

    assert result.volumetric_result is not None
    assert result.volumetric_result.liquid_damage_status == LiquidDamageStatus.CONFIRMED_LIQUID_DAMAGE
    assert FraudFlag.PHYSICAL_DAMAGE in result.fraud_flags
    assert result.disposition is not None
    # Verify itemized deduction was added for liquid damage
    has_liquid_deduction = any(
        "Liquid Ingress Damage" in d.reason for d in result.disposition.deduction_breakdown
    )
    assert has_liquid_deduction is True

