"""Unit and Integration Tests for the 4-Pillar Real-World Return Harvester."""

import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytest

from scripts.harvest_real_world_returns import (
    MANIFEST_PATH,
    PILLAR_1_TARGETS,
    PILLAR_2_TARGETS,
    PILLAR_3_STREAMS,
    PILLAR_4_TARGETS,
    PROJECT_ROOT,
    SAMPLE_RETURNS_DIR,
    RealWorldReturnHarvester,
)


def test_4_pillar_manifest_structure():
    """Validates that real_world_harvest_manifest.json adheres to 4-pillar schema."""
    assert MANIFEST_PATH.exists(), f"Manifest file missing at {MANIFEST_PATH}"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["version"] == "2.0.0"
    assert "4-Pillar" in data["strategy"]
    assert data["total_assets"] >= 30
    assert len(data["pillars"]) == 4

    records = data["records"]
    assert len(records) == data["total_assets"]

    pillars_found = set()
    for rec in records:
        rel_path = rec["path"]
        abs_path = PROJECT_ROOT / rel_path
        assert abs_path.exists(), f"Indexed asset does not exist on disk: {abs_path}"

        w, h = rec["width"], rec["height"]
        assert w > 0 and h > 0, f"Invalid dimensions {w}x{h} for {rel_path}"
        assert rec["file_size_bytes"] > 0
        assert rec["url"].startswith("http")
        assert "pillar" in rec
        assert "defect_type" in rec
        pillars_found.add(rec["pillar"])

    # Ensure all 4 pillars are represented in harvested assets
    assert len(pillars_found) == 4


def test_assembled_4_pillar_return_cases():
    """Verifies the 5 assembled real-world return cases in storage/sample_returns/."""
    expected_cases = [
        ("REAL-CASE-01-CLONE-SWAP-AIRREPS", "SKU-APPLE-APM-SLV", "FRAUD_HOLD"),
        ("REAL-CASE-02-SONY-CLONE-TEARDOWN", "SKU-SONY-WH1000XM5-BLK", "FRAUD_HOLD"),
        ("REAL-CASE-03-EBAY-BROKEN-HINGE", "SKU-SONY-WH1000XM5-BLK", "GRADE_C"),
        ("REAL-CASE-04-PALLET-UNBOXING-FRAME", "SKU-SONY-WH1000XM5-BLK", "GRADE_C"),
        ("REAL-CASE-05-CUSTOMER-EVIDENCE-MISSING-BOM", "SKU-APPLE-APM-SLV", "GRADE_D"),
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
        assert "pillar" in c_data

        chassis_img_path = case_dir / "chassis.jpg"
        assert chassis_img_path.exists(), f"Chassis photo missing for {case_id}"

        with Image.open(chassis_img_path) as img:
            assert img.size[0] > 100
            assert img.size[1] > 100


def test_video_keyframe_laplacian_sharpness():
    """Tests OpenCV frame sharpness computation on synthetic and real frames."""
    sharp_img = np.zeros((200, 200), dtype=np.uint8)
    sharp_img[::20, :] = 255
    sharp_img[:, ::20] = 255
    sharp_score = float(cv2.Laplacian(sharp_img, cv2.CV_64F).var())

    blurred_img = cv2.GaussianBlur(sharp_img, (25, 25), 0)
    blurred_score = float(cv2.Laplacian(blurred_img, cv2.CV_64F).var())

    assert sharp_score > blurred_score
    assert sharp_score > 500.0
    assert blurred_score < 200.0


def test_4_pillar_source_coverage():
    """Ensures targets cover replica communities, resale marketplaces, videos, and customer reviews."""
    assert len(PILLAR_1_TARGETS) >= 10
    assert any("airreps" in t["source"].lower() for t in PILLAR_1_TARGETS)
    assert any("walkman blog" in t["source"].lower() for t in PILLAR_1_TARGETS)

    assert len(PILLAR_2_TARGETS) >= 5
    assert any("resale" in t["category"].lower() for t in PILLAR_2_TARGETS)

    assert len(PILLAR_3_STREAMS) >= 2
    assert any("pallet" in s["description"].lower() for s in PILLAR_3_STREAMS)
    assert any("teardown" in s["description"].lower() for s in PILLAR_3_STREAMS)

    assert len(PILLAR_4_TARGETS) >= 4
    assert any("customer" in t["category"].lower() for t in PILLAR_4_TARGETS)

