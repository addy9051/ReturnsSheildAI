"""Tests for DatasetLoader and Real-World Benchmark Evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.catalog_manager import CatalogManager
from core.dataset_loader import DatasetLoader
from core.schemas import ROIAnnotation, SKUOnboardRequest, TriageGroundTruthAnnotation
from tests.synthetic_data import create_chassis_image


@pytest.fixture
def mock_annotated_dataset(tmp_path: Path) -> Path:
    """Creates a mock dataset conforming to TriageGroundTruthAnnotation."""
    dataset_dir = tmp_path / "benchmark_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create dummy overhead image
    img = np.full((600, 800, 3), fill_value=240, dtype=np.uint8)
    # Draw chassis box
    img[30:390, 30:390] = 40
    # Draw serial plate
    img[420:560, 420:760] = 50
    # Draw barcode label
    img[420:560, 40:380] = 250
    cv2.putText(img, "SERIAL: SN-SNY-8842109", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "S/N: SN-SNY-8842109", (430, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    img_path = dataset_dir / "item_001.png"
    cv2.imwrite(str(img_path), img)

    # 2. Create annotations.json
    annotation = {
        "image_id": "item_001",
        "product_sku": "SKU-SONY-WH1000XM5-BLK",
        "image_path": "item_001.png",
        "rois": [
            {
                "roi_id": "roi_serial",
                "class_id": "chassis_serial_plate",
                "bbox_xywh": [420, 420, 340, 140],
                "ocr_ground_truth": "SN-SNY-8842109",
                "attributes": {"material": "metallic_engraved"},
            },
            {
                "roi_id": "roi_barcode",
                "class_id": "box_barcode_label",
                "bbox_xywh": [40, 420, 340, 140],
                "ocr_ground_truth": "SN-SNY-8842109",
            },
            {
                "roi_id": "roi_chassis",
                "class_id": "product_chassis_main",
                "bbox_xywh": [30, 30, 360, 360],
            },
        ],
        "ground_truth_triage": {
            "fraud_flags": [],
            "is_fraud": False,
            "condition_grade": "GRADE_A",
            "disposition_action": "FULL_REFUND",
            "itemized_deductions": [],
            "net_refund_amount": 399.99,
        },
    }

    with open(dataset_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump([annotation], f, indent=2)

    return dataset_dir


def test_dataset_loader_parsing(mock_annotated_dataset: Path):
    """Verifies DatasetLoader correctly validates hierarchical schema records."""
    loader = DatasetLoader(mock_annotated_dataset)
    assert len(loader.records) == 1
    record = loader.records["item_001"]
    assert record.product_sku == "SKU-SONY-WH1000XM5-BLK"
    assert len(record.rois) == 3
    assert record.rois[0].ocr_ground_truth == "SN-SNY-8842109"


def test_dataset_loader_crop_roi(mock_annotated_dataset: Path):
    """Verifies crop_roi extracts sub-images with exact requested dimensions."""
    loader = DatasetLoader(mock_annotated_dataset)
    record = loader.records["item_001"]
    img_path = mock_annotated_dataset / record.image_path

    serial_roi = [r for r in record.rois if r.class_id == "chassis_serial_plate"][0]
    crop = loader.crop_roi(img_path, serial_roi)
    assert crop.shape == (140, 340, 3)


def test_dataset_benchmark_evaluation(mock_annotated_dataset: Path):
    """Verifies benchmark evaluation computes mathematical metrics."""
    loader = DatasetLoader(mock_annotated_dataset)
    metrics = loader.evaluate_benchmark()

    assert metrics["total_evaluated"] == 1
    assert "serial_verification_pass_rate" in metrics
    assert "mean_ocr_weighted_edit_distance" in metrics
    assert metrics["serial_verification_pass_rate"] == 1.0

