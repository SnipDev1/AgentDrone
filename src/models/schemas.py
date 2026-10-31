"""Pydantic data schemas for AgentDrone platform.

Defines strict typed contracts for drone telemetry, cadastral validation,
computer vision findings, and banking credit objects.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


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
    # Precision MRK metrology fields
    std_lat_m: float | None = None
    std_lon_m: float | None = None
    std_hgt_m: float | None = None
    horizontal_accuracy_m: float | None = None
    gps_week: int | None = None
    gps_time_sec: float | None = None
    antenna_offset_mm: tuple[float, float, float] | None = None
    # Anti-spoofing and telemetry consistency fields
    exif_latitude: float | None = None
    exif_longitude: float | None = None
    mrk_latitude: float | None = None
    mrk_longitude: float | None = None
    telemetry_discrepancy_m: float | None = None
    is_telemetry_consistent: bool = True


class PPKTelemetrySummary(BaseModel):
    """RINEX / PPK GNSS dual-frequency carrier-phase telemetry summary."""
    has_ppk: bool = False
    obs_filename: str = ""
    nav_filename: str = ""
    rinex_version: str = "3.05"
    constellations: list[str] = Field(default_factory=list)
    approx_satellites_tracked: int = 0
    start_time_utc: str = ""
    end_time_utc: str = ""
    duration_sec: float = 0.0
    dual_frequency_l1_l2: bool = True
    carrier_phase_kinematic: bool = True
    anti_spoofing_verified: bool = True
    geodetic_survey_grade: str = "Высокоточная геодезическая привязка (СКО < 5 см)"


class MultispectralAnalysis(BaseModel):
    """DJI Mavic 3M multispectral 4-band analysis (NIR, Red, Green, RedEdge)."""
    has_multispectral: bool = False
    mean_ndvi: float = 0.0
    construction_soil_pct: float = 0.0
    vegetation_pct: float = 0.0
    site_activity_status: str = "Норма"
    weed_overgrowth_risk: str = "Низкий"
    bands_available: list[str] = Field(default_factory=list)
    heatmap_rel_path: str | None = None


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
    ppk_summary: PPKTelemetrySummary | None = None
    multispectral: MultispectralAnalysis | None = None
    operator_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_reloaded_models(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for k, v in list(out.items()):
            if hasattr(v, "model_dump"):
                out[k] = v.model_dump()
            elif isinstance(v, list):
                out[k] = [item.model_dump() if hasattr(item, "model_dump") else item for item in v]
        return out


class CreditObject(BaseModel):
    """Bank credit registry object record (strictly matching customer CSV table)."""
    cadastral_id: str
    address: str = "Новосибирская область"
    planned_completion_date: str = "Нет"
    actual_completion_date: str = "Нет"
    has_pz: str = "Нет"
    is_registered: str = "Нет"
    is_pledged: str = "Нет"
    construction_start_date: str = "Нет"
    has_aerial_photo: str = "Да"
    is_minimal_schema: bool = True
    # Legacy optional fields retained with empty defaults for unit test compatibility only
    borrower_name: str = ""
    credit_agreement_number: str = ""
    credit_amount_rub: float = 0.0
    declared_wall_material: str = ""
    loan_purpose: str = "ИЖС"
    days_overdue: int = 0
    tranche_number: int = 1
    total_tranches: int = 3


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
