"""Ingestion module for drone telemetry, angles, and cadastral validation."""

from .angle_detector import AngleCompletenessDetector
from .cadastral_registry import (
    CadastralParcel,
    CadastralRegistry,
    normalize_cadastral_id,
)
from .cadastral_validator import (
    CadastralValidator,
    haversine_distance_m,
    project_camera_aim_ground,
)
from .credit_registry import CreditRegistry
from .notes_parser import NotesParser
from .ppk_parser import PPKParser
from .telemetry_parser import MRKRecord, TelemetryParser

__all__ = [
    "AngleCompletenessDetector",
    "CadastralParcel",
    "CadastralRegistry",
    "CadastralValidator",
    "CreditRegistry",
    "MRKRecord",
    "NotesParser",
    "PPKParser",
    "TelemetryParser",
    "haversine_distance_m",
    "normalize_cadastral_id",
    "project_camera_aim_ground",
]
