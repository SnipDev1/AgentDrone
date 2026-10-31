"""Domain models and schemas for AgentDrone.

Provides strongly-typed Pydantic models for credit contracts, drone inspections,
decision engine strategies, collateral inspection acts, and computer vision pipelines.
"""

# Domain models & Enums (Decision Engine, Reports, GigaChat, Registry)
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

# Computer Vision & Ingestion Schemas
from .schemas import (
    CadastralGeoCheck,
    CompletenessReport,
    CVInspectionResult,
    ShotAngleType,
    StructuralElements,
    DroneImageMeta as CVDroneImageMeta,
    CreditObject as CVCreditObject,
    WallMaterial as CVWallMaterial,
)

__all__ = [
    # Domain models
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
    # CV & Ingestion Schemas
    "CadastralGeoCheck",
    "CompletenessReport",
    "CVInspectionResult",
    "ShotAngleType",
    "StructuralElements",
    "CVDroneImageMeta",
    "CVCreditObject",
    "CVWallMaterial",
]
