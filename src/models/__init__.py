"""Domain models for AgentDrone.

Provides strongly-typed Pydantic models for credit contracts, drone inspections,
decision engine strategies, and collateral inspection acts.
"""

from .enums import (
    StageCategory,
    WallMaterial,
    BankStrategy,
    RiskLevel,
    CameraViewType,
    CollateralCondition,
)
from .credit import CreditObject
from .inspection import (
    DroneImageMeta,
    BuildingElements,
    DefectItem,
    InspectionResult,
)
from .strategy import (
    StrategyRecommendation,
    ReportPhotoItem,
    ActReportData,
)

__all__ = [
    "StageCategory",
    "WallMaterial",
    "BankStrategy",
    "RiskLevel",
    "CameraViewType",
    "CollateralCondition",
    "CreditObject",
    "DroneImageMeta",
    "BuildingElements",
    "DefectItem",
    "InspectionResult",
    "StrategyRecommendation",
    "ReportPhotoItem",
    "ActReportData",
]
