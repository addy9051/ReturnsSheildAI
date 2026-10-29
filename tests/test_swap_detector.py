"""Unit and Integration Tests for SwapDetector (Phase 2).

Tests:
1. DINOv2 cosine similarity & dense patch anomaly computation.
2. High-confidence genuine verification (VERIFIED_OEM).
3. Low-confidence swap detection (CONFIRMED_SWAP, SUSPECTED_SWAP).
4. Spatial diff heatmap generation, dimension matching, and file persistence.
5. Direct reference image comparison vs. catalog cache retrieval.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import pytest

from core.catalog_manager import CatalogManager
from core.schemas import IdentityMatchStatus, SKUOnboardRequest
from core.swap_detector import SwapDetector
from tests.synthetic_data import create_chassis_image


@pytest.fixture
def temp_storage(tmp_path: Path) -> Path:
    storage = tmp_path / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    return storage


@pytest.fixture
def catalog_with_sku(temp_storage: Path) -> CatalogManager:
    manager = CatalogManager(storage_dir=temp_storage)
    ref_img = create_chassis_image("authentic_black")
    manager.onboard_sku(
        SKUOnboardRequest(
            sku_id="SKU-XM5-BLACK",
            title="Sony WH-1000XM5 Black",
            category="Audio",
            msrp=399.99,
            similarity_threshold=0.85,
        ),
        reference_images=[ref_img],
    )
    return manager


def test_swap_detector_verified_oem(catalog_with_sku: CatalogManager, temp_storage: Path):
    """Verifies genuine return matches catalog reference with high similarity and low patch anomaly."""
    heatmaps_dir = temp_storage / "heatmaps"
    detector = SwapDetector(catalog_manager=catalog_with_sku, heatmaps_dir=heatmaps_dir)

    query_img = create_chassis_image("authentic_black")
    result = detector.detect_swap(query_img, sku_id="SKU-XM5-BLACK")

    assert result.identity_status == IdentityMatchStatus.VERIFIED_OEM
    assert result.is_swap_detected is False
    assert result.cosine_similarity >= 0.95
    assert result.spatial_diff_score <= 0.20
    assert result.heatmap_path is not None
    assert Path(result.heatmap_path).exists()


def test_swap_detector_confirmed_swap(catalog_with_sku: CatalogManager, temp_storage: Path):
    """Verifies cheap swapped model triggers CONFIRMED_SWAP fraud flag."""
    detector = SwapDetector(catalog_manager=catalog_with_sku)

    swapped_img = create_chassis_image("swapped_cheap_beige")
    result = detector.detect_swap(swapped_img, sku_id="SKU-XM5-BLACK")

    assert result.identity_status == IdentityMatchStatus.CONFIRMED_SWAP
    assert result.is_swap_detected is True
    assert result.cosine_similarity < 0.70
    assert result.spatial_diff_score >= 0.35
    assert "Bait-and-Switch Fraud Detected" in result.explanation


def test_swap_detector_neon_gaming_swap(catalog_with_sku: CatalogManager):
    """Verifies gaming headset swap is detected as fraud."""
    detector = SwapDetector(catalog_manager=catalog_with_sku)

    gaming_img = create_chassis_image("swapped_neon_gaming")
    result = detector.detect_swap(gaming_img, sku_id="SKU-XM5-BLACK")

    assert result.is_swap_detected is True
    assert result.cosine_similarity < 0.75


def test_spatial_heatmap_properties(catalog_with_sku: CatalogManager, temp_storage: Path):
    """Verifies generated heatmap overlay matches input dimensions and has RGB channels."""
    custom_heatmap = temp_storage / "test_heatmap.png"
    detector = SwapDetector(catalog_manager=catalog_with_sku)

    query_img = create_chassis_image("swapped_cheap_beige", size=(400, 400))
    result = detector.detect_swap(
        query_img,
        sku_id="SKU-XM5-BLACK",
        custom_heatmap_path=custom_heatmap,
    )

    assert Path(result.heatmap_path).resolve() == custom_heatmap.resolve()
    assert custom_heatmap.exists()

    with Image.open(custom_heatmap) as saved_img:
        assert saved_img.size == (400, 400)
        assert saved_img.mode in ("RGB", "RGBA")


def test_swap_detector_direct_reference(temp_storage: Path):
    """Verifies swap detection works with direct reference image without catalog lookup."""
    detector = SwapDetector(heatmaps_dir=temp_storage / "heatmaps")

    ref_img = create_chassis_image("authentic_black")
    query_img = create_chassis_image("authentic_black")

    result = detector.detect_swap(query_img, reference_image=ref_img)
    assert result.identity_status == IdentityMatchStatus.VERIFIED_OEM
    assert result.is_swap_detected is False


def test_swap_detector_unregistered_sku_error(temp_storage: Path):
    """Verifies appropriate KeyError is raised when unknown SKU is queried."""
    manager = CatalogManager(storage_dir=temp_storage)
    detector = SwapDetector(catalog_manager=manager)

    with pytest.raises(KeyError):
        detector.detect_swap(Image.new("RGB", (224, 224)), sku_id="NON_EXISTENT_SKU")

