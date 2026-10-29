"""Dual-Serial OCR Verifier Engine for ReturnShield AI.

Cross-matches outer packaging / RMA barcodes against laser-engraved
or printed chassis serial numbers. Resolves optical OCR confusion
(0/O, 1/I/L, 5/S, 2/Z) via Weighted Levenshtein distance and format regexes.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from core.schemas import SerialMatchStatus, SerialVerificationResult

logger = logging.getLogger("returnshield.serial_verifier")


class SerialVerifier:
    """Verifies box barcode against physical chassis serial number."""

    # Common OCR optical lookalike confusion pairs
    CONFUSION_PRIMARY: Set[Tuple[str, str]] = {
        ("O", "0"), ("0", "O"),
        ("I", "1"), ("1", "I"),
        ("L", "1"), ("1", "L"),
        ("I", "L"), ("L", "I"),
    }

    CONFUSION_SECONDARY: Set[Tuple[str, str]] = {
        ("S", "5"), ("5", "S"),
        ("B", "8"), ("8", "B"),
        ("Z", "2"), ("2", "Z"),
    }

    def __init__(
        self,
        default_regex: Optional[str] = r"^[A-Z0-9-]{8,20}$",
        barcode_detector: Optional[cv2.barcode.BarcodeDetector] = None,
    ):
        """Initializes barcode and OCR engines.

        Args:
            default_regex: Fallback regex pattern used when SKU profile does not define one.
            barcode_detector: Optional injected OpenCV BarcodeDetector instance.
        """
        self.default_regex = default_regex
        self._barcode_detector = barcode_detector
        self._ocr_engine = None

    @property
    def barcode_detector(self) -> cv2.barcode.BarcodeDetector:
        """Lazy-loads OpenCV native barcode detector."""
        if self._barcode_detector is None:
            self._barcode_detector = cv2.barcode.BarcodeDetector()
        return self._barcode_detector

    @property
    def ocr_engine(self):
        """Lazy-loads RapidOCR (PaddleOCR on ONNX) for scene character extraction."""
        if self._ocr_engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._ocr_engine = RapidOCR()
            except ImportError:
                logger.warning("RapidOCR not available; falling back to heuristic parsing.")
                self._ocr_engine = None
        return self._ocr_engine

    def preprocess_chassis_plate(self, bgr_image: np.ndarray) -> np.ndarray:
        """Applies CLAHE and bilateral filtering to enhance faint laser etching on metal.

        Why: Metallic chassis serial plates reflect warehouse ring lights, washing out
        engraved characters. CLAHE in LAB color space equalizes local luminance without
        introducing chromatic distortion.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Contrast Limited Adaptive Histogram Equalization on L channel
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)

        # Merge back and convert to BGR
        enhanced_lab = cv2.merge((cl, a_channel, b_channel))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Bilateral filter to smooth grain while preserving character edge sharpness
        denoised = cv2.bilateralFilter(enhanced_bgr, d=5, sigmaColor=50, sigmaSpace=50)
        return denoised

    def read_box_barcode(
        self,
        box_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> Tuple[Optional[str], float]:
        """Extracts barcode string from shipping label image or returns raw string.

        Returns:
            Tuple of (detected_barcode_string, confidence_score [0.0, 1.0]).
        """
        if isinstance(box_input, str) and not Path(box_input).exists():
            # Already a raw barcode string, not an image file path
            return box_input.strip(), 1.0

        # Load image
        if isinstance(box_input, (str, Path)):
            bgr = cv2.imread(str(box_input))
        elif isinstance(box_input, Image.Image):
            bgr = cv2.cvtColor(np.array(box_input), cv2.COLOR_RGB2BGR)
        elif isinstance(box_input, np.ndarray):
            bgr = box_input.copy()
        else:
            return None, 0.0

        if bgr is None:
            return None, 0.0

        # 1. Try OpenCV native barcode detector
        try:
            decode_res = self.barcode_detector.detectAndDecode(bgr)
            # OpenCV 5.0 returns (info, type, points) or (ret, info, type, points)
            info = decode_res[0] if isinstance(decode_res[0], str) else decode_res[1]
            if info and len(info.strip()) > 0:
                return info.strip(), 0.99
        except Exception as err:
            logger.debug("OpenCV barcode decode did not return 1D barcode: %s", err)

        # 2. Fallback: OCR reading of human-readable label text below barcode
        if self.ocr_engine:
            try:
                ocr_results, _ = self.ocr_engine(bgr)
                if ocr_results:
                    for item in ocr_results:
                        text = item[1].strip()
                        conf = float(item[2])
                        # Check for pattern like 'SERIAL: SN-SNY-8842109' or 'SN-SNY-8842109'
                        match = re.search(r"(?:SERIAL|S/N|SN)?[:\s]*([A-Z0-9-]{8,22})", text, re.IGNORECASE)
                        if match:
                            candidate = match.group(1).strip()
                            if len(candidate) >= 8 and any(c.isdigit() for c in candidate):
                                return candidate, conf
            except Exception as err:
                logger.error("OCR fallback on box barcode failed: %s", err)

        return None, 0.0

    def read_chassis_serial(
        self,
        chassis_input: Union[str, Path, Image.Image, np.ndarray],
        expected_regex: Optional[str] = None,
    ) -> Tuple[Optional[str], float]:
        """Extracts laser-engraved or stamped serial number from chassis plate.

        Returns:
            Tuple of (raw_chassis_serial_string, ocr_confidence [0.0, 1.0]).
        """
        if isinstance(chassis_input, str) and not Path(chassis_input).exists():
            # Direct string pass-through
            return chassis_input.strip(), 1.0

        if isinstance(chassis_input, (str, Path)):
            bgr = cv2.imread(str(chassis_input))
        elif isinstance(chassis_input, Image.Image):
            bgr = cv2.cvtColor(np.array(chassis_input), cv2.COLOR_RGB2BGR)
        elif isinstance(chassis_input, np.ndarray):
            bgr = chassis_input.copy()
        else:
            return None, 0.0

        if bgr is None:
            return None, 0.0

        # Preprocess plate image to minimize reflection washout
        preprocessed = self.preprocess_chassis_plate(bgr)

        if not self.ocr_engine:
            return None, 0.0

        try:
            ocr_results, _ = self.ocr_engine(preprocessed)
            if not ocr_results:
                # Retry on raw image if preprocessing was overly aggressive
                ocr_results, _ = self.ocr_engine(bgr)

            if not ocr_results:
                return None, 0.0

            regex_pattern = expected_regex or self.default_regex
            compiled_regex = re.compile(regex_pattern) if regex_pattern else None

            # First pass: look for exact regex match or 'S/N:' header line
            for item in ocr_results:
                text = item[1].strip()
                conf = float(item[2])

                # Check for explicit S/N prefix
                sn_match = re.search(r"(?:S/N|SERIAL|SN)[:\s]*([A-Z0-9-]{8,22})", text, re.IGNORECASE)
                if sn_match:
                    candidate = sn_match.group(1).strip()
                    return candidate, conf

                # Check against SKU regex
                if compiled_regex and compiled_regex.match(text):
                    return text, conf

            # Second pass: heuristic alphanumeric candidate with digits and letters
            candidates: List[Tuple[str, float]] = []
            for item in ocr_results:
                text = item[1].strip().replace(" ", "")
                conf = float(item[2])
                if len(text) >= 8 and any(c.isdigit() for c in text) and any(c.isalpha() for c in text):
                    candidates.append((text, conf))

            if candidates:
                # Return highest confidence candidate
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[0]

        except Exception as err:
            logger.error("Chassis serial OCR failed: %s", err)

        return None, 0.0

    @classmethod
    def compute_weighted_distance(cls, str1: str, str2: str) -> float:
        """Computes dynamic-programming Levenshtein distance with discounted OCR lookalike costs."""
        s1 = str1.upper()
        s2 = str2.upper()
        m, n = len(s1), len(s2)

        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = float(i)
        for j in range(n + 1):
            dp[0][j] = float(j)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                c1, c2 = s1[i - 1], s2[j - 1]
                if c1 == c2:
                    sub_cost = 0.0
                elif (c1, c2) in cls.CONFUSION_PRIMARY:
                    sub_cost = 0.15
                elif (c1, c2) in cls.CONFUSION_SECONDARY:
                    sub_cost = 0.25
                else:
                    sub_cost = 1.0

                dp[i][j] = min(
                    dp[i - 1][j] + 1.0,       # Deletion
                    dp[i][j - 1] + 1.0,       # Insertion
                    dp[i - 1][j - 1] + sub_cost  # Substitution
                )

        return dp[m][n]

    @classmethod
    def normalize_serial(cls, raw: Optional[str]) -> str:
        """Cleans formatting noise while preserving alphanumeric characters and hyphens."""
        if not raw:
            return ""
        # Strip trailing/leading spaces and common prefixes
        cleaned = raw.strip().upper()
        cleaned = re.sub(r"^(?:S/N|SERIAL|SN)[:\s-]*", "", cleaned, flags=re.IGNORECASE)
        # Normalize internal whitespace
        cleaned = re.sub(r"\s+", "", cleaned)
        return cleaned

    def verify_serials(
        self,
        box_barcode_input: Union[str, Path, Image.Image, np.ndarray, None],
        chassis_serial_input: Union[str, Path, Image.Image, np.ndarray, None],
        expected_regex: Optional[str] = None,
    ) -> SerialVerificationResult:
        """Executes dual-serial cross-verification between outer box and product chassis.

        Args:
            box_barcode_input: Box shipping barcode image or decoded string.
            chassis_serial_input: Chassis serial plate crop image or OCR string.
            expected_regex: Expected serial number regex pattern for format validation.

        Returns:
            SerialVerificationResult with match status, similarity ratio, and OCR confidence.
        """
        # 1. Extract box barcode
        box_raw: Optional[str] = None
        box_conf = 1.0
        if box_barcode_input is not None:
            box_raw, box_conf = self.read_box_barcode(box_barcode_input)

        # 2. Extract chassis serial
        chassis_raw: Optional[str] = None
        chassis_conf = 1.0
        if chassis_serial_input is not None:
            chassis_raw, chassis_conf = self.read_chassis_serial(chassis_serial_input, expected_regex)

        # 3. Handle unreadable or missing cases
        if not box_raw and not chassis_raw:
            return SerialVerificationResult(
                box_barcode=None,
                chassis_serial_raw=None,
                chassis_serial_normalized=None,
                match_status=SerialMatchStatus.MISSING_LABEL,
                is_match=False,
                similarity_ratio=0.0,
                ocr_confidence=0.0,
                notes="Neither box barcode nor chassis serial plate could be read.",
            )
        elif not box_raw:
            return SerialVerificationResult(
                box_barcode=None,
                chassis_serial_raw=chassis_raw,
                chassis_serial_normalized=self.normalize_serial(chassis_raw),
                match_status=SerialMatchStatus.MISSING_LABEL,
                is_match=False,
                similarity_ratio=0.0,
                ocr_confidence=chassis_conf,
                notes="Box barcode is missing or unreadable.",
            )
        elif not chassis_raw:
            return SerialVerificationResult(
                box_barcode=box_raw,
                chassis_serial_raw=None,
                chassis_serial_normalized=None,
                match_status=SerialMatchStatus.OCR_UNREADABLE,
                is_match=False,
                similarity_ratio=0.0,
                ocr_confidence=0.0,
                notes="Chassis serial plate is unreadable, scratched, or missing.",
            )

        # 4. Normalize extracted strings
        box_norm = self.normalize_serial(box_raw)
        chassis_norm = self.normalize_serial(chassis_raw)

        # 5. Compute Weighted Levenshtein Distance
        weighted_dist = self.compute_weighted_distance(box_norm, chassis_norm)
        max_len = max(len(box_norm), len(chassis_norm), 1)
        similarity_ratio = max(0.0, min(1.0, 1.0 - (weighted_dist / float(max_len))))

        # 6. Classify match status
        if box_norm == chassis_norm:
            status = SerialMatchStatus.EXACT_MATCH
            is_match = True
            notes = f"Exact character match verified: '{box_norm}'."
        elif weighted_dist <= 0.60 and similarity_ratio >= 0.85:
            # Discounted optical ambiguity (e.g. O/0, 1/I, S/5 confusion across up to 2-3 characters)
            status = SerialMatchStatus.NORMALIZED_MATCH
            is_match = True
            notes = (
                f"Normalized match verified after OCR lookalike ambiguity resolution "
                f"(Box: '{box_norm}', Chassis: '{chassis_norm}', WeightedDist: {weighted_dist:.2f}, Sim: {similarity_ratio:.3f})."
            )
        else:
            status = SerialMatchStatus.MISMATCH
            is_match = False
            notes = (
                f"Serial Number Mismatch (Potential Fraud): Box barcode '{box_norm}' "
                f"does not match chassis serial '{chassis_norm}' (WeightedDist: {weighted_dist:.2f})."
            )

        logger.info(
            "Serial verification: %s (Box='%s', Chassis='%s', Sim=%.3f)",
            status.value, box_norm, chassis_norm, similarity_ratio
        )

        return SerialVerificationResult(
            box_barcode=box_raw,
            chassis_serial_raw=chassis_raw,
            chassis_serial_normalized=chassis_norm,
            match_status=status,
            is_match=is_match,
            similarity_ratio=round(similarity_ratio, 4),
            ocr_confidence=round(chassis_conf, 4),
            notes=notes,
        )
