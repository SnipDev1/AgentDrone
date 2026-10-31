"""Pydantic data schemas for AgentDrone platform.

Defines strict typed contracts for drone telemetry, cadastral validation,
computer vision findings, and banking credit objects.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ShotAngleType(str, Enum):
    """Standard drone shot angles according to Bank inspection guidelines."""
    FACADE_NORTH = "Фасад (Север)"
    FACADE_SOUTH = "Фасад (Юг)"
    FACADE_EAST = "Фасад (Восток)"
    FACADE_WEST = "Фасад (Запад)"
    NADIR_LOW = "Надир детальный (<30м)"
    NADIR_HIGH = "Надир обзорный (>=30м)"
    OBLIQUE_OTHER = "Перспективный ракурс"
    NADIR_OTHER = "Вертикальный кадр"
    UNKNOWN = "Неопределенный ракурс"


class WallMaterial(str, Enum):
    """Recognized wall construction materials."""
    BRICK = "Кирпич"
    GAS_CONCRETE = "Газобетон/Блок"
    TIMBER = "Брус/Дерево"
    FRAME_PANEL = "Щитовой каркас/SIP"
    SANDWICH_PANEL = "Сэндвич-панели/Коммерческий"
    UNKNOWN = "Не определен"


class DroneImageMeta(BaseModel):
    """Telemetry and orientation metadata extracted from a single drone shot."""
    filename: str
    file_path: str
    timestamp: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    altitude_rel_m: float | None = None
    altitude_abs_m: float | None = None
    gimbal_pitch_deg: float | None = None
    gimbal_yaw_deg: float | None = None
    gimbal_roll_deg: float | None = None
    flight_yaw_deg: float | None = None
    rtk_flag: int | None = None
    rtk_status: str | None = None
    shot_angle: ShotAngleType = ShotAngleType.UNKNOWN
    shutter_index: int | None = None
    drone_model: str | None = None
    is_dji: bool = True


class CompletenessReport(BaseModel):
    """Verification of mandatory camera angles (4 facades + 2 nadirs)."""
    total_shots: int = 0
    facades_count: int = 0
    nadir_count: int = 0
    has_4_facades: bool = False
    has_2_nadir: bool = False
    is_fully_compliant: bool = False
    present_angles: list[str] = Field(default_factory=list)
    missing_angles: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CadastralGeoCheck(BaseModel):
    """Cadastral geofencing and coordinate verification."""
    cadastral_id: str
    raw_folder_name: str = ""
    drone_center_lat: float
    drone_center_lon: float
    registry_lat: float | None = None
    registry_lon: float | None = None
    registry_address: str | None = None
    registry_vri: str | None = None
    distance_to_cadastre_m: float = 0.0
    mean_camera_aim_dist_m: float = 0.0
    max_distance_from_center_m: float = 0.0
    is_geofence_valid: bool = True
    within_allowed_radius: bool = True
    is_correct_plot: bool = True
    verification_status: str = "ПОДТВЕРЖДЕНО"
    notes: str = "Координаты подтверждены"


class StructuralElements(BaseModel):
    """Detected building structural components."""
    has_foundation: bool = False
    has_walls: bool = False
    has_roof: bool = False
    has_windows_doors: bool = False
    foundation_confidence: float = 0.0
    walls_confidence: float = 0.0
    roof_confidence: float = 0.0
    windows_doors_confidence: float = 0.0


class CVInspectionResult(BaseModel):
    """Consolidated computer vision analysis for an inspection plot."""
    cadastral_id: str
    readiness_pct: float = Field(..., ge=0.0, le=100.0, description="Физический процент готовности (0..100%)")
    readiness_stage: int = Field(..., description="Нормативная банковская стадия (0, 50, 100%)")
    stage_title: str = Field(..., description="Наименование стадии по регламенту Банка")
    structural_elements: StructuralElements = Field(default_factory=StructuralElements)
    detected_material: WallMaterial = WallMaterial.UNKNOWN
    material_confidence: float = 0.0
    completeness: CompletenessReport = Field(default_factory=CompletenessReport)
    geocheck: CadastralGeoCheck
    fraud_risk_flag: bool = False
    fraud_reasons: list[str] = Field(default_factory=list)
    defects_detected: list[str] = Field(default_factory=list)
    recommendation_hint: str = ""
    analyzed_images_count: int = 0
    shots_metadata: list[DroneImageMeta] = Field(default_factory=list)


class CreditObject(BaseModel):
    """Bank credit registry object record."""
    cadastral_id: str
    borrower_name: str
    credit_agreement_number: str
    credit_amount_rub: float
    planned_completion_date: str
    declared_wall_material: str
    loan_purpose: str = "ИЖС"
    days_overdue: int = 0
    tranche_number: int = 1
    total_tranches: int = 3
    address: str = "Новосибирская область"


class StrategyType(str, Enum):
    """5 Core Banking Credit Monitoring Strategies."""
    APPROVE_TRANCHE = "Выдача очередного транша"
    RESTRUCTURE_EXTEND = "Реструктуризация / Продление срока"
    REGISTER_COLLATERAL = "Регистрация залога / Закрытие ПЗ"
    SUSPEND_SECURITY_FRAUD = "Блокировка / Служба безопасности"
    DEFAULT_FORECLOSURE = "Дефолт / Взыскание залога"


class StrategyRecommendation(BaseModel):
    """Actionable recommendation generated by the Bank Decision Engine."""
    strategy: StrategyType
    strategy_code: str  # e.g. "STRAT-01".."STRAT-05"
    title: str
    risk_level: str  # "LOW", "MEDIUM", "CRITICAL"
    action_required: str
    reasoning: list[str] = Field(default_factory=list)
    financial_impact_rub: float = 0.0
    legal_basis: str = ""
    kalita_status: str = "PENDING"
