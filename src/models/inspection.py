"""Pydantic models for drone telemetry metadata and CV inspection results.

Contains schemas for individual image orientation/GNSS, detected structural
elements, defects, and overall readiness calculations.
"""

from datetime import datetime, date
from typing import List, Optional
from pydantic import BaseModel, Field

from .enums import StageCategory, WallMaterial, CameraViewType


class DroneImageMeta(BaseModel):
    """Metadata extracted from DJI EXIF / Timestamp.MRK for a single photo."""
    filename: str = Field(..., description="Имя файла изображения")
    filepath: Optional[str] = Field(default=None, description="Относительный или абсолютный путь к файлу")
    timestamp: Optional[datetime] = Field(default=None, description="Метка времени съемки")
    lat: Optional[float] = Field(default=None, description="Широта съемки WGS84")
    lon: Optional[float] = Field(default=None, description="Долгота съемки WGS84")
    alt_ellipsoid: Optional[float] = Field(default=None, description="Высота относительно эллипсоида (м)")
    gimbal_pitch: Optional[float] = Field(
        default=None,
        description="Угол наклона подвеса камеры (-90° надир, -45° перспектива)"
    )
    gimbal_yaw: Optional[float] = Field(default=None, description="Азимут камеры (0..360°)")
    view_type: CameraViewType = Field(
        default=CameraViewType.UNKNOWN,
        description="Классифицированный ракурс снимка"
    )
    rtk_status: Optional[str] = Field(default=None, description="Флаг точности GNSS (50 - RTK Fix)")
    std_lat: Optional[float] = Field(default=None, description="СКО по широте (м)")
    std_lon: Optional[float] = Field(default=None, description="СКО по долготе (м)")
    std_alt: Optional[float] = Field(default=None, description="СКО по высоте (м)")


class BuildingElements(BaseModel):
    """Structural elements detected during CV inspection."""
    foundation: bool = Field(default=False, description="Наличие фундамента / цоколя")
    walls: bool = Field(default=False, description="Наличие несущих стен")
    roof: bool = Field(default=False, description="Наличие смонтированной кровли")
    windows_doors: bool = Field(default=False, description="Установка оконных и дверных блоков (закрытый контур)")
    scaffolding_or_cranes: bool = Field(default=False, description="Признаки активной стройки (леса, техника)")


class DefectItem(BaseModel):
    """Structural defect or violation observed during inspection."""
    defect_type: str = Field(..., description="Тип дефекта (напр. 'Отсутствие консервации', 'Критические трещины')")
    severity: str = Field(default="MEDIUM", description="Критичность: LOW, MEDIUM, HIGH, CRITICAL")
    description: str = Field(..., description="Подробное описание дефекта")
    recommendation: Optional[str] = Field(default=None, description="Рекомендация по устранению")


class InspectionResult(BaseModel):
    """Aggregated CV & drone inspection analysis for a cadastral plot."""
    cadastral_number: str = Field(..., description="Кадастровый номер участка")
    inspection_date: date = Field(..., description="Дата проведения аэрофотосъемки")
    total_images_analyzed: int = Field(default=0, ge=0, description="Количество обработанных снимков")

    # Регламентные проверки съемки
    view_angles_complete: bool = Field(
        default=False,
        description="Соблюдение ТЗ: минимум 4 перспективных фасада + 2 верхних кадра"
    )
    geo_match: bool = Field(
        default=True,
        description="Геовалидация: точки съемки находятся в пределах участка (<50 м от границы)"
    )
    distance_to_boundary_m: Optional[float] = Field(
        default=None,
        description="Фактическое расстояние от стен до межи участка (норматив >= 3.0 м)"
    )
    boundary_violation: bool = Field(
        default=False,
        description="Флаг нарушения градостроительного отступа 3м (риск ст. 222 ГК РФ)"
    )

    # Строительная готовность
    elements: BuildingElements = Field(
        default_factory=BuildingElements,
        description="Распознанные конструктивные элементы"
    )
    physical_readiness_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Непрерывный расчетный процент готовности строения (0..100%)"
    )
    stage: StageCategory = Field(
        default=StageCategory.STAGE_0,
        description="Нормативная категория готовности Банка (0% / 50% / 100%)"
    )

    # Анализ материалов и целевого использования
    detected_wall_material: WallMaterial = Field(
        default=WallMaterial.UNKNOWN,
        description="Фактический материал стен по результатам компьютерного зрения"
    )
    material_match: bool = Field(
        default=True,
        description="Совпадение фактического материала с проектной сметой заемщика"
    )
    is_commercial_building_risk: bool = Field(
        default=False,
        description="Признаки нецелевого использования (коммерческий склад / магазин вместо ИЖС)"
    )

    # Дефекты и консервация
    defects: List[DefectItem] = Field(default_factory=list, description="Список зафиксированных дефектов")
    is_construction_abandoned: bool = Field(
        default=False,
        description="Признаки заброшенности стройки без консервации (>6 мес. без динамики)"
    )
