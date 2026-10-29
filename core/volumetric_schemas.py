"""Volumetric and Multi-Angle Stereo Inspection Schemas for ReturnShield AI.

Defines schemas for:
- Camera perspectives (Top-down 90 deg, Oblique 45 deg, Stereo Left/Right).
- Volumetric depth profiling and dimensional bounding envelopes.
- Liquid Contact / Damage Indicator (LDI) detection.
- Lateral hinge fracture and I/O port integrity findings.
- Multi-view feature fusion verdicts.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field


class CameraPerspective(str, Enum):
    """Camera viewpoint relative to inspection station tray."""
    TOP_DOWN = "TOP_DOWN"          # 90-degree orthogonal overhead view
    OBLIQUE_45 = "OBLIQUE_45"      # 45-degree lateral perspective view
    STEREO_LEFT = "STEREO_LEFT"    # Calibrated stereo left baseline
    STEREO_RIGHT = "STEREO_RIGHT"  # Calibrated stereo right baseline


class PortCondition(str, Enum):
    """Condition of I/O connectors, USB-C ports, and headphone jacks."""
    PRISTINE = "PRISTINE"
    DIRT_OBSTRUCTED = "DIRT_OBSTRUCTED"
    PINS_BENT_DAMAGED = "PINS_BENT_DAMAGED"
    BURN_CORROSION = "BURN_CORROSION"
    UNKNOWN = "UNKNOWN"


class HingeCondition(str, Enum):
    """Structural condition of lateral pivot joints and hinges."""
    INTACT_OEM = "INTACT_OEM"
    FRACTURED_CRACKED = "FRACTURED_CRACKED"
    LOOSE_DISLOCATED = "LOOSE_DISLOCATED"
    UNAUTHORIZED_GLUE_REPAIR = "UNAUTHORIZED_GLUE_REPAIR"


class LiquidDamageStatus(str, Enum):
    """Status of physical Liquid Damage Indicator (LDI) stickers."""
    NO_LIQUID_DETECTED = "NO_LIQUID_DETECTED"    # White / Silver pristine indicator
    SUSPECTED_MOISTURE = "SUSPECTED_MOISTURE"    # Faint pink gradient
    CONFIRMED_LIQUID_DAMAGE = "CONFIRMED_LIQUID_DAMAGE"  # Crimson / Dark Red triggered indicator


class VolumetricDiscrepancy(BaseModel):
    """Volumetric 3D profile discrepancy against SKU OEM specifications."""
    model_config = ConfigDict(extra="ignore")

    measured_depth_mm: float = Field(..., ge=0.0)
    nominal_depth_mm: float = Field(..., ge=0.0)
    depth_delta_mm: float
    tolerance_mm: float = Field(default=5.0)
    discrepancy_detected: bool = Field(default=False)
    disparity_mean: float = Field(default=0.0)
    disparity_std: float = Field(default=0.0)


class MultiAngleInspectionResult(BaseModel):
    """Comprehensive findings from multi-angle oblique & volumetric stereo analysis."""
    model_config = ConfigDict(extra="ignore")

    inspection_id: str = Field(default_factory=lambda: f"VOL-{uuid.uuid4().hex[:10].upper()}")
    liquid_damage_status: LiquidDamageStatus = Field(default=LiquidDamageStatus.NO_LIQUID_DETECTED)
    liquid_indicator_pixels: int = Field(default=0, ge=0)
    port_condition: PortCondition = Field(default=PortCondition.PRISTINE)
    hinge_condition: HingeCondition = Field(default=HingeCondition.INTACT_OEM)
    volumetric_profile: Optional[VolumetricDiscrepancy] = None
    disparity_map_path: Optional[str] = None
    multi_view_fused_similarity: float = Field(default=1.0, ge=-1.0, le=1.0)
    top_similarity: float = Field(default=1.0, ge=-1.0, le=1.0)
    oblique_similarity: float = Field(default=1.0, ge=-1.0, le=1.0)
    lateral_tamper_detected: bool = Field(default=False)
    lateral_findings: List[str] = Field(default_factory=list)

