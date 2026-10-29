"""Unit and Integration Tests for SerialVerifier (Phase 2).

Tests:
1. End-to-end OCR extraction on box barcode and chassis serial plate crops.
2. Optical character confusion resolution (0/O, 1/I/L, 5/S, 2/Z).
3. Genuine serial number mismatch detection.
4. Handling of missing or unreadable labels.
5. Image preprocessing (CLAHE in LAB color space).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.schemas import SerialMatchStatus
from core.serial_verifier import SerialVerifier


@pytest.fixture
def verifier() -> SerialVerifier:
    return SerialVerifier()


def test_serial_verifier_exact_match(verifier: SerialVerifier):
    """Verifies identical strings yield EXACT_MATCH with 1.0 similarity."""
    result = verifier.verify_serials("SN-SNY-8842109", "SN-SNY-8842109")
    assert result.match_status == SerialMatchStatus.EXACT_MATCH
    assert result.is_match is True
    assert result.similarity_ratio == 1.0


def test_serial_verifier_ocr_lookalike_confusion(verifier: SerialVerifier):
    """Verifies optical noise (O for 0, I for 1) resolves to NORMALIZED_MATCH without false fraud flag."""
    # Case A: 'O' instead of '0', and 'I' instead of '1'
    result = verifier.verify_serials("SN-SNY-8842109", "SN-SNY-8842IO9")
    assert result.match_status == SerialMatchStatus.NORMALIZED_MATCH
    assert result.is_match is True
    assert result.similarity_ratio >= 0.95

    # Case B: 'L' instead of '1'
    result_l = verifier.verify_serials("SN-SNY-8842109", "SN-SNY-8842L09")
    assert result_l.match_status == SerialMatchStatus.NORMALIZED_MATCH
    assert result_l.is_match is True

    # Case C: 'S' instead of '5'
    result_s = verifier.verify_serials("SN-DEV-5544", "SN-DEV-SS44")
    assert result_s.match_status == SerialMatchStatus.NORMALIZED_MATCH
    assert result_s.is_match is True


def test_serial_verifier_genuine_mismatch(verifier: SerialVerifier):
    """Verifies swapped chassis unit with completely different serial is flagged as MISMATCH."""
    result = verifier.verify_serials("SN-SNY-8842109", "SN-SNY-1102934")
    assert result.match_status == SerialMatchStatus.MISMATCH
    assert result.is_match is False
    assert result.similarity_ratio < 0.60
    assert "Serial Number Mismatch" in result.notes


def test_serial_verifier_missing_inputs(verifier: SerialVerifier):
    """Verifies missing or unreadable inputs produce proper error statuses."""
    # Missing box barcode
    res_no_box = verifier.verify_serials(None, "SN-SNY-8842109")
    assert res_no_box.match_status == SerialMatchStatus.MISSING_LABEL
    assert res_no_box.is_match is False

    # Missing chassis serial
    res_no_chassis = verifier.verify_serials("SN-SNY-8842109", None)
    assert res_no_chassis.match_status == SerialMatchStatus.OCR_UNREADABLE
    assert res_no_chassis.is_match is False

    # Neither provided
    res_none = verifier.verify_serials(None, None)
    assert res_none.match_status == SerialMatchStatus.MISSING_LABEL
    assert res_none.is_match is False


def test_serial_normalizer(verifier: SerialVerifier):
    """Tests whitespace and prefix normalization."""
    assert verifier.normalize_serial("  S/N: SN-SNY-8842109  ") == "SN-SNY-8842109"
    assert verifier.normalize_serial("SERIAL: ABC-1234") == "ABC-1234"
    assert verifier.normalize_serial("sn: 9944-1122") == "9944-1122"


def test_clahe_preprocessing(verifier: SerialVerifier):
    """Tests that CLAHE enhancement preserves dimensions and color channels."""
    dummy_plate = np.full((120, 300, 3), fill_value=60, dtype=np.uint8)
    enhanced = verifier.preprocess_chassis_plate(dummy_plate)
    assert enhanced.shape == (120, 300, 3)
    assert enhanced.dtype == np.uint8


def test_image_crops_e2e_verification(verifier: SerialVerifier):
    """Tests end-to-end verification using actual disk image crops from CASE-01-CLEAN."""
    box_crop = Path("storage/sample_returns/CASE-01-CLEAN/box_barcode_crop.png")
    serial_crop = Path("storage/sample_returns/CASE-01-CLEAN/serial_plate_crop.png")

    if box_crop.exists() and serial_crop.exists():
        result = verifier.verify_serials(box_crop, serial_crop)
        assert result.is_match is True
        assert result.match_status in (SerialMatchStatus.EXACT_MATCH, SerialMatchStatus.NORMALIZED_MATCH)
        assert result.box_barcode is not None
        assert result.chassis_serial_raw is not None

