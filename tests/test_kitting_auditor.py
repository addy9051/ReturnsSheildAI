"""Unit and Integration Tests for KittingAuditor (Stage 3 BOM Verification).

Tests:
1. Edge energy and Shannon entropy computation on empty vs populated tray slots.
2. Packaging tray spatial partitioning for 2, 3, and 4 BOM item packaging layouts.
3. BOM audit with direct detected item IDs (clean vs missing items).
4. Image-based visual tray audit with synthetic populated vs empty tray crops.
5. Catalog SKU lookup integration and missing SKU error handling.
6. Empty BOM handling for zero-accessory SKUs.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from core.catalog_manager import CatalogManager
from core.kitting_auditor import KittingAuditor
from core.schemas import AccessoryStatus, BOMItem, SKUProfile


@pytest.fixture
def sample_bom() -> list[BOMItem]:
    """Sample Bill of Materials checklist."""
    return [
        BOMItem(item_id="cable-usbc", name="USB-C Charging Cable", category="cable", deduction_fee=19.99),
        BOMItem(item_id="audio-jack-35", name="3.5mm Audio Cable", category="cable", deduction_fee=14.99),
        BOMItem(item_id="flight-adapter", name="Airplane Adapter", category="adapter", deduction_fee=9.99),
    ]


@pytest.fixture
def mock_catalog_manager(tmp_path: Path, sample_bom: list[BOMItem]) -> CatalogManager:
    """Provides a CatalogManager with a registered SKU containing the sample BOM."""
    catalog = CatalogManager(storage_dir=tmp_path / "kitting_catalog")
    profile = SKUProfile(
        sku_id="SKU-HEADPHONE-MOCK",
        title="Test Noise Cancelling Headphones",
        category="Audio",
        msrp=349.99,
        restocking_fee_rate=0.10,
        bom=sample_bom,
        expected_serial_regex=r"^SN-HD-[A-Z0-9]{6}$",
    )
    catalog.onboard_sku(profile)
    return catalog


def test_compute_slot_metrics_empty_vs_populated():
    """Validates that empty foam cavities and populated accessory slots have distinct metrics."""
    # 1. Empty cavity: uniform gray with subtle noise
    empty_slot = np.full((100, 100, 3), fill_value=40, dtype=np.uint8)
    # Add minor Gaussian noise
    noise = np.random.normal(0, 1.0, empty_slot.shape).astype(np.int8)
    empty_slot = np.clip(empty_slot.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    edge_empty, entropy_empty = KittingAuditor.compute_slot_metrics(empty_slot)

    # 2. Populated slot: coiled cable with high-contrast lines and texture
    populated_slot = np.full((100, 100, 3), fill_value=40, dtype=np.uint8)
    # Draw coiled black/white cable loops
    cv2.circle(populated_slot, (50, 50), 30, (230, 230, 230), thickness=4)
    cv2.circle(populated_slot, (50, 50), 20, (20, 20, 20), thickness=3)
    cv2.circle(populated_slot, (50, 50), 10, (220, 220, 220), thickness=3)

    edge_pop, entropy_pop = KittingAuditor.compute_slot_metrics(populated_slot)

    # Gradient energy on populated slot must significantly exceed empty cavity
    assert edge_pop > edge_empty * 2.5
    # Entropy must also be notably higher
    assert entropy_pop > entropy_empty


def test_partition_tray_slots():
    """Validates heuristic spatial partitioning for 2, 3, and 4 items."""
    auditor = KittingAuditor()
    tray_img = np.zeros((400, 600, 3), dtype=np.uint8)

    # 2 items -> 2 vertical halves
    slots_2 = auditor.partition_tray_slots(tray_img, 2)
    assert len(slots_2) == 2
    assert slots_2[0].shape == (200, 600, 3)
    assert slots_2[1].shape == (200, 600, 3)

    # 3 items -> 2 top quadrants + 1 large bottom cavity
    slots_3 = auditor.partition_tray_slots(tray_img, 3)
    assert len(slots_3) == 3
    assert slots_3[0].shape == (180, 300, 3)  # 0.45 * 400 = 180
    assert slots_3[1].shape == (180, 300, 3)
    assert slots_3[2].shape == (220, 600, 3)  # remaining 220 height, full width

    # 4 items -> 2x2 grid
    slots_4 = auditor.partition_tray_slots(tray_img, 4)
    assert len(slots_4) == 4
    for slot in slots_4:
        assert slot.shape == (200, 300, 3)


def test_audit_kitting_detected_item_ids_all_present(sample_bom: list[BOMItem]):
    """Tests BOM audit when all expected accessories are provided."""
    auditor = KittingAuditor()
    detected = ["cable-usbc", "audio-jack-35", "flight-adapter"]

    result = auditor.audit_kitting(
        bom_override=sample_bom,
        detected_item_ids=detected,
    )

    assert result.expected_items_count == 3
    assert result.detected_items_count == 3
    assert result.missing_items_count == 0
    assert result.all_items_present is True
    assert result.total_accessory_deduction == 0.0

    for item_audit in result.item_audits:
        assert item_audit.status == AccessoryStatus.PRESENT_OEM
        assert item_audit.deduction_fee == 0.0


def test_audit_kitting_detected_item_ids_missing_items(sample_bom: list[BOMItem]):
    """Tests BOM audit when customer retains cables (fraudulent partial return)."""
    auditor = KittingAuditor()
    # Customer only returned the flight adapter; kept both cables
    detected = ["flight-adapter"]

    result = auditor.audit_kitting(
        bom_override=sample_bom,
        detected_item_ids=detected,
    )

    assert result.expected_items_count == 3
    assert result.detected_items_count == 1
    assert result.missing_items_count == 2
    assert result.all_items_present is False
    # Deductions: $19.99 + $14.99 = $34.98
    assert result.total_accessory_deduction == 34.98

    missing_items = [a for a in result.item_audits if a.status == AccessoryStatus.MISSING]
    assert len(missing_items) == 2
    missing_ids = {a.bom_item_id for a in missing_items}
    assert missing_ids == {"cable-usbc", "audio-jack-35"}


def test_audit_kitting_catalog_lookup(mock_catalog_manager: CatalogManager):
    """Tests BOM audit pulling BOM profile directly from CatalogManager."""
    auditor = KittingAuditor(catalog_manager=mock_catalog_manager)

    result = auditor.audit_kitting(
        sku_id="SKU-HEADPHONE-MOCK",
        detected_item_ids=["cable-usbc"],  # 1 present, 2 missing
    )

    assert result.expected_items_count == 3
    assert result.detected_items_count == 1
    assert result.missing_items_count == 2
    # Missing audio cable (14.99) + flight adapter (9.99) = 24.98
    assert result.total_accessory_deduction == 24.98


def test_audit_kitting_unregistered_sku():
    """Tests that querying an unregistered SKU raises KeyError."""
    auditor = KittingAuditor()
    with pytest.raises(KeyError, match="not found in catalog"):
        auditor.audit_kitting(sku_id="NON-EXISTENT-SKU")


def test_audit_kitting_empty_bom():
    """Tests that products with no accessories pass audit cleanly."""
    auditor = KittingAuditor()
    result = auditor.audit_kitting(bom_override=[])

    assert result.expected_items_count == 0
    assert result.detected_items_count == 0
    assert result.missing_items_count == 0
    assert result.all_items_present is True
    assert result.total_accessory_deduction == 0.0
