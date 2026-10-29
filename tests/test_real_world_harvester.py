"""Unit and Integration Tests for Real-World Return Harvester."""

import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytest

from scripts.harvest_real_world_returns import (
    IFIXIT_GUIDES,
    IFIXIT_WIKIS,
    MANIFEST_PATH,
    PROJECT_ROOT,
    REAL_CATALOG_DIR,
    SAMPLE_RETURNS_DIR,
    WIKIMEDIA_COMMONS_ITEMS,
    RealWorldReturnHarvester,
)


def test_manifest_file_structure():
    """Validates that real_world_harvest_manifest.json exists and conforms to schema."""
    assert MANIFEST_PATH.exists(), f"Manifest file missing at {MANIFEST_PATH}"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["version"] == "1.0.0"
    assert data["total_assets"] >= 50
    assert "iFixit Guides" in data["sources"]
    assert "Wikimedia Commons" in data["sources"]
    assert "Video Stream Keyframes" in data["sources"]

    records = data["records"]
    assert len(records) == data["total_assets"]

    # Verify every indexed file exists on disk with valid image dimensions
    for rec in records:
        rel_path = rec["path"]
        abs_path = PROJECT_ROOT / rel_path
        assert abs_path.exists(), f"Indexed asset does not exist on disk: {abs_path}"

        w, h = rec["width"], rec["height"]
        assert w > 0 and h > 0, f"Invalid dimensions {w}x{h} for {rel_path}"
        assert rec["file_size_bytes"] > 0
        assert rec["url"].startswith("http")
        assert "license" in rec


def test_assembled_real_return_cases():
    """Verifies the 5 assembled real-world return cases in storage/sample_returns/."""
    expected_cases = [
        ("REAL-CASE-01-SONY-BROKEN-HINGE", "SKU-SONY-WH1000XM5-BLK", "GRADE_C"),
        ("REAL-CASE-02-AIRPODS-MAX-DISASSEMBLED", "SKU-APPLE-APM-SLV", "GRADE_D"),
        ("REAL-CASE-03-BOSE-QC-AUTHENTIC-KIT", "SKU-BOSE-QCU-BLK", "GRADE_A"),
        ("REAL-CASE-04-DYSON-BROKEN-VACUUM", "SKU-DYSON-V15-DET", "GRADE_C"),
        ("REAL-CASE-05-SWAP-MDR-FOR-XM5", "SKU-SONY-WH1000XM5-BLK", "FRAUD_HOLD"),
    ]

    for case_id, expected_sku, expected_grade in expected_cases:
        case_dir = SAMPLE_RETURNS_DIR / case_id
        assert case_dir.is_dir(), f"Case directory missing: {case_dir}"

        manifest_file = case_dir / "manifest.json"
        assert manifest_file.exists(), f"Case manifest missing in {case_dir}"

        with open(manifest_file, "r", encoding="utf-8") as f:
            c_data = json.load(f)

        assert c_data["case_id"] == case_id
        assert c_data["sku_id"] == expected_sku
        assert c_data["expected_condition"] == expected_grade

        chassis_img_path = case_dir / "chassis.jpg"
        assert chassis_img_path.exists(), f"Chassis photo missing for {case_id}"

        # Confirm valid PIL image
        with Image.open(chassis_img_path) as img:
            assert img.size[0] > 100
            assert img.size[1] > 100


def test_video_keyframe_laplacian_sharpness():
    """Tests OpenCV frame sharpness computation on synthetic and real frames."""
    # Create sharp checkerboard pattern
    sharp_img = np.zeros((200, 200), dtype=np.uint8)
    sharp_img[::20, :] = 255
    sharp_img[:, ::20] = 255
    sharp_score = float(cv2.Laplacian(sharp_img, cv2.CV_64F).var())

    # Create blurry image
    blurred_img = cv2.GaussianBlur(sharp_img, (25, 25), 0)
    blurred_score = float(cv2.Laplacian(blurred_img, cv2.CV_64F).var())

    assert sharp_score > blurred_score
    assert sharp_score > 500.0
    assert blurred_score < 200.0


def test_curated_source_coverage():
    """Ensures curated guide and asset lists cover all high-value product categories."""
    skus_in_guides = {g["sku_id"] for g in IFIXIT_GUIDES}
    assert "SKU-SONY-WH1000XM5-BLK" in skus_in_guides
    assert "SKU-APPLE-APM-SLV" in skus_in_guides
    assert "SKU-BOSE-QCU-BLK" in skus_in_guides
    assert "SKU-DYSON-V15-DET" in skus_in_guides

    skus_in_wikis = {w["sku_id"] for w in IFIXIT_WIKIS}
    assert len(skus_in_wikis) >= 4

    wiki_commons_cases = {w["use_case"] for w in WIKIMEDIA_COMMONS_ITEMS}
    assert "BAIT_AND_SWITCH_IMPOSTOR" in wiki_commons_cases
    assert "AUTHENTIC_BOM_CARRYCASE" in wiki_commons_cases
    assert "DAMAGED_RETURN_VACUUM" in wiki_commons_cases
