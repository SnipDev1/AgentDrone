"""Ingestion module for drone telemetry, angles, and cadastral validation."""

from .angle_detector import AngleCompletenessDetector
from .cadastral_validator import CadastralValidator, haversine_distance_m
from .telemetry_parser import MRKRecord, TelemetryParser

__all__ = [
    "AngleCompletenessDetector",
    "CadastralValidator",
    "MRKRecord",
    "TelemetryParser",
    "haversine_distance_m",
]
