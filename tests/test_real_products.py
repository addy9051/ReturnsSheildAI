"""Integration Tests for Real-World Product Catalog Ingestion and Swap Verification."""

from __future__ import annotations

from pathlib import Path
import pytest

from core.catalog_manager import CatalogManager
from core.schemas import IdentityMatchStatus
from core.swap_detector import SwapDetector


@pytest.fixture(scope="module")
def catalog() -> CatalogManager:
    return CatalogManager()


@pytest.fixture(scope="module")
def swap_detector(catalog: CatalogManager) -> SwapDetector:
    return SwapDetector(catalog_manager=catalog)


def test_real_catalog_contains_curated_skus(catalog: CatalogManager):
    """Verifies that all 5 real commercial products are registered with valid embeddings."""
    expected_skus = [
        "SKU-SONY-WH1000XM5-BLK",
        "SKU-BOSE-QCU-BLK",
        "SKU-APPLE-APM-SLV",
        "SKU-DEWALT-DCD800B",
        "SKU-DYSON-V15-DET",
    ]
    for sku_id in expected_skus:
        profile = catalog.get_sku(sku_id)
        assert profile is not None, f"SKU {sku_id} missing from catalog"
        assert profile.msrp > 100.0
        assert len(profile.bom) >= 3

        emb = catalog.get_sku_embedding(sku_id)
        assert emb is not None
        assert emb.shape == (384,)

        patch_emb = catalog.get_sku_patch_embeddings(sku_id)
        assert patch_emb is not None
        assert patch_emb.shape == (256, 384)


def test_real_product_self_verification(catalog: CatalogManager, swap_detector: SwapDetector):
    """Verifies that authentic commercial catalog photos pass OEM verification."""
    skus_to_test = ["SKU-BOSE-QCU-BLK", "SKU-APPLE-APM-SLV", "SKU-DEWALT-DCD800B"]
    for sku_id in skus_to_test:
        profile = catalog.get_sku(sku_id)
        assert profile is not None
        img_path = profile.reference_image_paths[0]
        assert Path(img_path).exists()

        result = swap_detector.detect_swap(img_path, sku_id=sku_id, save_heatmap=False)
        assert result.identity_status == IdentityMatchStatus.VERIFIED_OEM
        assert result.is_swap_detected is False
        assert result.cosine_similarity >= 0.95


def test_cross_brand_headphones_swap_attack(catalog: CatalogManager, swap_detector: SwapDetector):
    """Tests subtle lookalike swap: customer returns Bose QC Ultra in Sony XM5 box."""
    bose_profile = catalog.get_sku("SKU-BOSE-QCU-BLK")
    assert bose_profile is not None
    bose_img = bose_profile.reference_image_paths[0]

    # Query against Sony XM5 SKU
    result = swap_detector.detect_swap(bose_img, sku_id="SKU-SONY-WH1000XM5-BLK", save_heatmap=False)
    assert result.is_swap_detected is True
    assert result.identity_status in (IdentityMatchStatus.SUSPECTED_SWAP, IdentityMatchStatus.CONFIRMED_SWAP)
    # Patch anomaly should be elevated
    assert result.spatial_diff_score >= 0.45


def test_cross_category_swap_attack(catalog: CatalogManager, swap_detector: SwapDetector):
    """Tests blatant counterfeit/wrong item swap: power drill in vacuum box."""
    dewalt_profile = catalog.get_sku("SKU-DEWALT-DCD800B")
    assert dewalt_profile is not None
    dewalt_img = dewalt_profile.reference_image_paths[0]

    result = swap_detector.detect_swap(dewalt_img, sku_id="SKU-DYSON-V15-DET", save_heatmap=False)
    assert result.identity_status == IdentityMatchStatus.CONFIRMED_SWAP
    assert result.is_swap_detected is True
    assert result.cosine_similarity < 0.30

