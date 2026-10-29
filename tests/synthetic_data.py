"""Synthetic Return Test Case Generator for ReturnShield AI.

Generates realistic mock return test cases including:
1. Clean Return (Authentic unit, matching serials, complete BOM)
2. Bait-and-Switch Fraud (Swapped low-value/wrong model or colorway)
3. Serial Mismatch Fraud (Authentic model chassis swapped with another unit / sticker peel)
4. Missing BOM Accessory (Missing charging cable / power adapter)
5. Cosmetic Wear / Damage (Authentic unit with heavy surface wear)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from core.schemas import (
    AccessoryStatus,
    BOMItem,
    ConditionGrade,
    DispositionAction,
    FraudFlag,
    SerialMatchStatus,
    SKUProfile,
)


@dataclass
class MockReturnCase:
    """Represents a fully articulated synthetic return inspection test scenario."""
    case_id: str
    title: str
    sku_id: str
    rma_number: str
    description: str
    box_barcode: str
    chassis_serial: str
    detected_accessories: List[str]
    expected_fraud_flags: List[FraudFlag]
    expected_disposition: DispositionAction
    expected_condition: ConditionGrade
    expected_serial_status: SerialMatchStatus
    surface_wear_severity: float  # 0.0 (pristine) to 1.0 (destroyed)
    overhead_image_path: Path
    chassis_crop_path: Path
    serial_plate_crop_path: Path
    tray_crop_path: Path
    oblique_image_path: Optional[Path] = None


def create_oblique_image(
    variant: str = "authentic_black",
    liquid_damage: bool = False,
    flank_swap: bool = False,
    fractured_hinge: bool = False,
    size: Tuple[int, int] = (400, 400),
) -> Image.Image:
    """Generates synthetic 45-degree oblique perspective camera view of product flank and ports."""
    if flank_swap:
        # Swapped flank housing (e.g. vintage beige or mismatched material)
        img = Image.new("RGB", size, color=(225, 220, 205))
        draw = ImageDraw.Draw(img)
        draw.ellipse([80, 80, 320, 320], fill=(210, 205, 190), outline=(170, 165, 150), width=4)
        draw.text((120, 190), "SWAPPED FLANK", fill=(100, 95, 80))
        return img

    img = Image.new("RGB", size, color=(235, 238, 240))  # warehouse mat background
    draw = ImageDraw.Draw(img)

    # Lateral headphone earcup and pivot bracket
    draw.ellipse([60, 90, 340, 330], fill=(42, 45, 48), outline=(65, 68, 72), width=3)
    # Headband slider hinge bracket
    draw.rounded_rectangle([170, 30, 230, 120], radius=8, fill=(35, 38, 40), outline=(60, 62, 65), width=2)

    # 3.5mm recessed audio port cavity
    draw.ellipse([185, 280, 215, 310], fill=(15, 15, 18), outline=(80, 82, 85), width=2)

    if liquid_damage:
        # Triggered crimson / pink Liquid Damage Indicator (LDI) inside port cavity
        draw.ellipse([193, 288, 207, 302], fill=(230, 25, 35))
    else:
        # Pristine silver/white untriggered LDI sticker
        draw.ellipse([193, 288, 207, 302], fill=(240, 242, 245))

    if fractured_hinge:
        # High-contrast jagged fracture line across hinge
        draw.line([(180, 50), (200, 75), (185, 95), (220, 110)], fill=(255, 255, 255), width=3)

    return img


def create_chassis_image(
    variant: str = "authentic_black",
    scratches: bool = False,
    size: Tuple[int, int] = (400, 400),
) -> Image.Image:
    """Generates synthetic product chassis image."""
    img = Image.new("RGB", size, color=(235, 238, 240))  # warehouse mat background
    draw = ImageDraw.Draw(img)

    if variant == "authentic_black":
        # Realistic matte black headphone silhouette
        # Headband
        draw.arc([100, 60, 300, 260], start=180, end=0, fill=(35, 36, 38), width=24)
        # Left and right earcups
        draw.rounded_rectangle([70, 160, 140, 310], radius=25, fill=(40, 42, 45), outline=(20, 20, 20), width=2)
        draw.rounded_rectangle([260, 160, 330, 310], radius=25, fill=(40, 42, 45), outline=(20, 20, 20), width=2)
        # Golden logo accent
        draw.rectangle([88, 175, 122, 185], fill=(212, 175, 55))
        draw.rectangle([278, 175, 312, 185], fill=(212, 175, 55))

    elif variant == "swapped_cheap_beige":
        # Drastically different shape & color (e.g. cheap beige vintage wired headphones)
        draw.arc([110, 80, 290, 240], start=180, end=0, fill=(180, 160, 130), width=14)
        draw.ellipse([60, 180, 150, 290], fill=(215, 200, 175), outline=(130, 110, 80), width=3)
        draw.ellipse([250, 180, 340, 290], fill=(215, 200, 175), outline=(130, 110, 80), width=3)
        # Red coiled cord
        draw.line([105, 290, 105, 370], fill=(200, 30, 30), width=6)

    elif variant == "swapped_neon_gaming":
        # Swapped neon green/cyan gaming headset
        draw.arc([90, 50, 310, 270], start=180, end=0, fill=(15, 240, 140), width=28)
        draw.polygon([(60, 170), (145, 150), (145, 320), (60, 300)], fill=(20, 20, 25), outline=(15, 240, 140), width=4)
        draw.polygon([(340, 170), (255, 150), (255, 320), (340, 300)], fill=(20, 20, 25), outline=(15, 240, 140), width=4)
        # Boom mic
        draw.line([120, 290, 190, 330], fill=(15, 240, 140), width=8)

    else:
        # Default placeholder shape
        draw.rectangle([100, 100, 300, 300], fill=(80, 80, 80))

    if scratches:
        # Draw visible silver/white abrasion marks and scratches
        for offset in [0, 15, 30]:
            draw.line([90 + offset, 210, 120 + offset, 270], fill=(220, 220, 220), width=2)
            draw.line([270 + offset, 220, 310 + offset, 255], fill=(220, 220, 220), width=2)
        draw.line([85, 240, 135, 245], fill=(240, 240, 240), width=3)

    return img


def create_serial_plate_image(
    serial_text: str,
    tampered: bool = False,
    size: Tuple[int, int] = (300, 120),
) -> Image.Image:
    """Generates synthetic cropped chassis serial engraving/label image."""
    img = Image.new("RGB", size, color=(50, 52, 55))  # Dark chassis metal/matte finish
    draw = ImageDraw.Draw(img)

    # Silver printed serial label boundary
    draw.rounded_rectangle([15, 15, size[0] - 15, size[1] - 15], radius=6, outline=(140, 145, 150), width=2)

    # Simulated micro-print text header
    draw.text((25, 22), "MODEL: WH-1000XM5  |  MADE IN MALAYSIA", fill=(180, 185, 190))
    draw.text((25, 42), "RATED: 5V = 2.0A", fill=(160, 165, 170))

    # Serial number line (crisp or blurred/scratched)
    draw.text((25, 68), f"S/N: {serial_text}", fill=(245, 245, 250))

    if tampered:
        # Draw peeling corner and deep scratches over serial
        draw.polygon([(size[0] - 50, 15), (size[0] - 15, 50), (size[0] - 15, 15)], fill=(30, 30, 30))
        draw.line([40, 75, 260, 75], fill=(20, 20, 20), width=4)
        draw.line([70, 65, 200, 85], fill=(20, 20, 20), width=3)

    return img


def create_box_barcode_image(
    barcode_text: str,
    size: Tuple[int, int] = (320, 120),
) -> Image.Image:
    """Generates synthetic box shipping / RMA label barcode."""
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw label border
    draw.rectangle([4, 4, size[0] - 4, size[1] - 4], outline=(180, 180, 180), width=1)
    draw.text((15, 10), "ORIGINAL PACKAGING BARCODE", fill=(80, 80, 80))

    # Deterministic barcode stripe generator from barcode string
    x = 25
    rng = np.random.RandomState(abs(hash(barcode_text)) % (2**31))
    while x < size[0] - 30:
        bar_w = int(rng.choice([1, 2, 3, 4]))
        draw.rectangle([x, 30, x + bar_w, 85], fill=(0, 0, 0))
        gap = int(rng.choice([2, 3, 4, 5]))
        x += bar_w + gap

    draw.text((45, 93), f"SERIAL: {barcode_text}", fill=(10, 10, 10))
    return img


def create_tray_image(
    cable_present: bool = True,
    case_present: bool = True,
    manual_present: bool = True,
    size: Tuple[int, int] = (400, 400),
) -> Image.Image:
    """Generates synthetic accessory tray ROI."""
    img = Image.new("RGB", size, color=(210, 212, 215))  # Gray foam tray
    draw = ImageDraw.Draw(img)

    # Molded recessed cavities
    # Slot 1: USB-C Cable (Top right slot)
    draw.rounded_rectangle([40, 40, 180, 180], radius=12, fill=(180, 182, 186), outline=(140, 142, 145), width=2)
    if cable_present:
        # Coiled black USB-C cable
        draw.ellipse([60, 60, 160, 160], outline=(25, 25, 25), width=10)
        draw.rectangle([105, 50, 120, 75], fill=(200, 200, 200))  # USB-C plug head
    else:
        draw.text((65, 100), "[EMPTY SLOT]", fill=(120, 120, 125))

    # Slot 2: 3.5mm Audio Cable (Top left slot)
    draw.rounded_rectangle([220, 40, 360, 180], radius=12, fill=(180, 182, 186), outline=(140, 142, 145), width=2)
    if manual_present:
        # Folded quick start guide / warranty booklet
        draw.rectangle([240, 60, 340, 160], fill=(250, 250, 250), outline=(200, 200, 200), width=1)
        draw.text((255, 100), "MANUAL", fill=(80, 80, 80))
    else:
        draw.text((245, 100), "[EMPTY SLOT]", fill=(120, 120, 125))

    # Slot 3: Hard Protective Carrying Case (Bottom large slot)
    draw.rounded_rectangle([40, 210, 360, 370], radius=16, fill=(180, 182, 186), outline=(140, 142, 145), width=2)
    if case_present:
        draw.rounded_rectangle([60, 225, 340, 355], radius=18, fill=(70, 72, 75), outline=(40, 40, 42), width=3)
        # Zipper track
        draw.arc([70, 235, 330, 345], start=0, end=180, fill=(180, 180, 180), width=3)
    else:
        draw.text((150, 280), "[EMPTY CAVITY]", fill=(120, 120, 125))

    return img


def create_overhead_composite(
    chassis_img: Image.Image,
    tray_img: Image.Image,
    barcode_img: Image.Image,
    serial_plate_img: Image.Image,
    size: Tuple[int, int] = (800, 600),
) -> Image.Image:
    """Composites product chassis, tray, and barcodes into a single bench capture."""
    composite = Image.new("RGB", size, color=(245, 247, 248))

    # Paste components into triage grid
    # Left: Product Chassis
    composite.paste(chassis_img.resize((360, 360)), (30, 30))
    # Right: Accessory Tray
    composite.paste(tray_img.resize((340, 340)), (420, 30))
    # Bottom Left: Box Barcode Label
    composite.paste(barcode_img.resize((340, 140)), (40, 420))
    # Bottom Right: Chassis Serial Plate Crop
    composite.paste(serial_plate_img.resize((340, 140)), (420, 420))

    # Draw triage bench guidelines
    draw = ImageDraw.Draw(composite)
    draw.text((32, 12), "ZONE 1: PRODUCT CHASSIS ROI", fill=(100, 100, 100))
    draw.text((422, 12), "ZONE 2: ACCESSORY TRAY ROI", fill=(100, 100, 100))
    draw.text((42, 402), "ZONE 3: OUTER BOX BARCODE", fill=(100, 100, 100))
    draw.text((422, 402), "ZONE 4: CHASSIS SERIAL PLATE", fill=(100, 100, 100))

    return composite


def generate_synthetic_return_dataset(
    output_base_dir: Path,
    sku_id: str = "SKU-SONY-WH1000XM5-BLK",
) -> Dict[str, MockReturnCase]:
    """Generates all mock return scenarios and persists imagery to disk.

    Returns:
        Dictionary mapping case_id to MockReturnCase.
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    cases: Dict[str, MockReturnCase] = {}
    genuine_serial = "SN-SNY-8842109"

    # Define the 5 core scenarios
    scenarios = [
        {
            "case_id": "CASE-01-CLEAN",
            "title": "Clean Return (Genuine Like-New)",
            "description": "Legitimate open-box return. Authentic black XM5, matching serials, all accessories present.",
            "chassis_variant": "authentic_black",
            "scratches": False,
            "box_barcode": genuine_serial,
            "chassis_serial": genuine_serial,
            "tampered_serial": False,
            "cable": True,
            "case": True,
            "manual": True,
            "detected_accessories": ["cable-usbc", "case-travel", "doc-manual"],
            "expected_fraud_flags": [FraudFlag.CLEAN],
            "expected_disposition": DispositionAction.FULL_REFUND,
            "expected_condition": ConditionGrade.GRADE_A,
            "expected_serial_status": SerialMatchStatus.EXACT_MATCH,
            "wear_severity": 0.0,
        },
        {
            "case_id": "CASE-02-SWAP",
            "title": "Bait-and-Switch Fraud (Swapped Cheap Model)",
            "description": "Customer returned cheap beige vintage headphones in place of the $399 Sony XM5.",
            "chassis_variant": "swapped_cheap_beige",
            "scratches": False,
            "box_barcode": genuine_serial,
            "chassis_serial": "SWAP-CHEAPO-991",
            "tampered_serial": True,
            "cable": False,
            "case": False,
            "manual": False,
            "detected_accessories": [],
            "expected_fraud_flags": [FraudFlag.BAIT_AND_SWITCH, FraudFlag.SERIAL_MISMATCH],
            "expected_disposition": DispositionAction.REJECT_FRAUD_HOLD,
            "expected_condition": ConditionGrade.FRAUD_HOLD,
            "expected_serial_status": SerialMatchStatus.MISMATCH,
            "wear_severity": 0.8,
        },
        {
            "case_id": "CASE-03-SERIAL-MISMATCH",
            "title": "Serial Number Mismatch (Old Unit Returned in New Box)",
            "description": "Customer bought new XM5 and returned their older broken unit in the new box.",
            "chassis_variant": "authentic_black",
            "scratches": False,
            "box_barcode": genuine_serial,
            "chassis_serial": "SN-SNY-1102934",  # Discrepant serial
            "tampered_serial": False,
            "cable": True,
            "case": True,
            "manual": True,
            "detected_accessories": ["cable-usbc", "case-travel", "doc-manual"],
            "expected_fraud_flags": [FraudFlag.SERIAL_MISMATCH],
            "expected_disposition": DispositionAction.REJECT_FRAUD_HOLD,
            "expected_condition": ConditionGrade.FRAUD_HOLD,
            "expected_serial_status": SerialMatchStatus.MISMATCH,
            "wear_severity": 0.05,
        },
        {
            "case_id": "CASE-04-MISSING-ACCESSORIES",
            "title": "Missing BOM Accessory (Cables Kept by Buyer)",
            "description": "Genuine XM5 returned with matching serial, but USB-C cable and carrying case are missing.",
            "chassis_variant": "authentic_black",
            "scratches": False,
            "box_barcode": genuine_serial,
            "chassis_serial": genuine_serial,
            "tampered_serial": False,
            "cable": False,  # Missing
            "case": False,   # Missing
            "manual": True,
            "detected_accessories": ["doc-manual"],
            "expected_fraud_flags": [FraudFlag.MISSING_ACCESSORY],
            "expected_disposition": DispositionAction.PARTIAL_REFUND,
            "expected_condition": ConditionGrade.GRADE_A,
            "expected_serial_status": SerialMatchStatus.EXACT_MATCH,
            "wear_severity": 0.0,
        },
        {
            "case_id": "CASE-05-COSMETIC-WEAR",
            "title": "Heavy Surface Wear / Scratches (Grade B / C)",
            "description": "Genuine unit returned with deep scratches across headband and earcups.",
            "chassis_variant": "authentic_black",
            "scratches": True,
            "box_barcode": genuine_serial,
            "chassis_serial": genuine_serial,
            "tampered_serial": False,
            "cable": True,
            "case": True,
            "manual": True,
            "detected_accessories": ["cable-usbc", "case-travel", "doc-manual"],
            "expected_fraud_flags": [FraudFlag.PHYSICAL_DAMAGE],
            "expected_disposition": DispositionAction.PARTIAL_REFUND,
            "expected_condition": ConditionGrade.GRADE_C,
            "expected_serial_status": SerialMatchStatus.EXACT_MATCH,
            "wear_severity": 0.45,
            "liquid_damage": False,
            "flank_swap": False,
        },
        {
            "case_id": "CASE-06-LIQUID-DAMAGE",
            "title": "Liquid Ingress Damage (LDI Triggered)",
            "description": "Genuine XM5 returned with triggered crimson Liquid Damage Indicator in lateral port cavity.",
            "chassis_variant": "authentic_black",
            "scratches": False,
            "box_barcode": genuine_serial,
            "chassis_serial": genuine_serial,
            "tampered_serial": False,
            "cable": True,
            "case": True,
            "manual": True,
            "detected_accessories": ["cable-usbc", "case-travel", "doc-manual"],
            "expected_fraud_flags": [FraudFlag.PHYSICAL_DAMAGE],
            "expected_disposition": DispositionAction.PARTIAL_REFUND,
            "expected_condition": ConditionGrade.GRADE_C,
            "expected_serial_status": SerialMatchStatus.EXACT_MATCH,
            "wear_severity": 0.1,
            "liquid_damage": True,
            "flank_swap": False,
        },
        {
            "case_id": "CASE-07-COVERT-SWAP",
            "title": "Covert Sandwich Swap (Top OEM vs Flank Swap)",
            "description": "Fraudster maintained genuine top headband shell but swapped internal/underbelly housing.",
            "chassis_variant": "authentic_black",
            "scratches": False,
            "box_barcode": genuine_serial,
            "chassis_serial": genuine_serial,
            "tampered_serial": False,
            "cable": True,
            "case": True,
            "manual": True,
            "detected_accessories": ["cable-usbc", "case-travel", "doc-manual"],
            "expected_fraud_flags": [FraudFlag.BAIT_AND_SWITCH],
            "expected_disposition": DispositionAction.REJECT_FRAUD_HOLD,
            "expected_condition": ConditionGrade.FRAUD_HOLD,
            "expected_serial_status": SerialMatchStatus.EXACT_MATCH,
            "wear_severity": 0.4,
            "liquid_damage": False,
            "flank_swap": True,
        },
    ]

    for item in scenarios:
        case_id = item["case_id"]
        case_dir = output_base_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        # Generate individual images
        chassis_img = create_chassis_image(
            variant=item["chassis_variant"],
            scratches=item["scratches"],
        )
        chassis_path = case_dir / "chassis_crop.png"
        chassis_img.save(chassis_path, "PNG")

        serial_plate_img = create_serial_plate_image(
            serial_text=item["chassis_serial"],
            tampered=item["tampered_serial"],
        )
        serial_path = case_dir / "serial_plate_crop.png"
        serial_plate_img.save(serial_path, "PNG")

        barcode_img = create_box_barcode_image(
            barcode_text=item["box_barcode"],
        )
        barcode_path = case_dir / "box_barcode_crop.png"
        barcode_img.save(barcode_path, "PNG")

        tray_img = create_tray_image(
            cable_present=item["cable"],
            case_present=item["case"],
            manual_present=item["manual"],
        )
        tray_path = case_dir / "tray_crop.png"
        tray_img.save(tray_path, "PNG")

        oblique_img = create_oblique_image(
            variant=item["chassis_variant"],
            liquid_damage=item.get("liquid_damage", False),
            flank_swap=item.get("flank_swap", False),
        )
        oblique_path = case_dir / "oblique_crop.png"
        oblique_img.save(oblique_path, "PNG")

        # Composite overhead triage bench snapshot
        overhead_img = create_overhead_composite(
            chassis_img=chassis_img,
            tray_img=tray_img,
            barcode_img=barcode_img,
            serial_plate_img=serial_plate_img,
        )
        overhead_path = case_dir / "overhead_snapshot.png"
        overhead_img.save(overhead_path, "PNG")

        case_obj = MockReturnCase(
            case_id=case_id,
            title=item["title"],
            sku_id=sku_id,
            rma_number=f"RMA-2026-{case_id.split('-')[1]}",
            description=item["description"],
            box_barcode=item["box_barcode"],
            chassis_serial=item["chassis_serial"],
            detected_accessories=item["detected_accessories"],
            expected_fraud_flags=item["expected_fraud_flags"],
            expected_disposition=item["expected_disposition"],
            expected_condition=item["expected_condition"],
            expected_serial_status=item["expected_serial_status"],
            surface_wear_severity=item["wear_severity"],
            overhead_image_path=overhead_path,
            chassis_crop_path=chassis_path,
            serial_plate_crop_path=serial_path,
            tray_crop_path=tray_path,
            oblique_image_path=oblique_path,
        )
        cases[case_id] = case_obj

    return cases

