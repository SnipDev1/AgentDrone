"""Pydantic schemas for multi-temporal construction dynamics and flight history.

Enables tracking construction progress across multiple drone inspection dates
for the same cadastral parcel, computing deltas, pace of work, and driving AI Agent analysis.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .schemas import WallMaterial


class FlightSnapshot(BaseModel):
    """Snapshot of a single drone flight inspection epoch for a cadastral parcel."""

    flight_id: str = Field(..., description="Идентификатор вылета (напр. 2026-05-15)")
    flight_date: str = Field(..., description="Дата проведения съемки (ГГГГ-ММ-ДД)")
    readiness_pct: float = Field(..., ge=0.0, le=100.0, description="Фактическая готовность (0..100%)")
    stage: int = Field(default=0, description="Нормативная стадия Банка (0, 50, 100%)")
    stage_title: str = Field(default="Стадия 0%", description="Наименование стадии")
    has_foundation: bool = False
    has_walls: bool = False
    has_roof: bool = False
    has_windows_doors: bool = False
    detected_material: WallMaterial = WallMaterial.UNKNOWN
    is_material_overridden: bool = False
    shots_count: int = 0
    shots_paths: list[str] = Field(default_factory=list)
    defects: list[str] = Field(default_factory=list)
    operator_notes: list[str] = Field(default_factory=list)


class ConstructionDynamics(BaseModel):
    """Comparative multi-temporal analysis of construction dynamics across inspection dates."""

    cadastral_id: str = Field(..., description="Кадастровый номер земельного участка")
    snapshots: list[FlightSnapshot] = Field(
        default_factory=list,
        description="Хронологический список снимков инспекций (от раннего к позднему)",
    )
    has_multiple_dates: bool = Field(default=False, description="Признак наличия нескольких дат съемки")
    delta_readiness_pct: float = Field(default=0.0, description="Прирост готовности между съемками (%)")
    delta_days: int = Field(default=0, description="Количество дней между съемками")
    pace_pct_per_month: float = Field(default=0.0, description="Среднемесячный темп СМР (% в месяц)")
    pace_status: str = Field(default="В графике", description="Оценка темпа (Опережение / В графике / Отставание / Заморозка)")
    elements_progress: list[str] = Field(
        default_factory=list,
        description="Перечень возведенных конструктивных элементов за прошедший период",
    )
    ai_agent_summary: str = Field(
        default="",
        description="Экспертное резюме ИИ-агента GigaChat по динамике строительства для АС Калита",
    )
    ai_agent_is_online: bool = Field(default=False, description="Статус генерации (Онлайн GigaChat / Оффлайн)")
