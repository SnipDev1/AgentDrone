"""Pydantic model for credit agreement and cadastral plot data.

Represents an individual housing construction (ИЖС) loan record
from Siberian Bank (Sber) loan portfolio.
"""

from datetime import date
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from .enums import WallMaterial


class CreditObject(BaseModel):
    """Credit agreement and cadastral collateral specification."""
    credit_id: str = Field(..., description="ID кредитного договора")
    cadastral_number: str = Field(
        ...,
        description="Кадастровый номер участка (например, 54:07:055001:178 или 5407055001178)"
    )
    borrower_name: str = Field(..., description="ФИО заемщика")
    credit_amount: float = Field(..., gt=0, description="Общая сумма кредитной линии (руб.)")
    current_disbursed: float = Field(default=0.0, ge=0, description="Фактически выданный объем средств (руб.)")
    tranche_number: int = Field(default=1, ge=1, le=10, description="Текущий этап / транш (1..10)")
    tranches_total: int = Field(default=3, ge=1, le=10, description="Всего траншей по договору")
    next_tranche_amount: float = Field(default=0.0, ge=0, description="Сумма к выдаче следующего транша (руб.)")
    credit_issue_date: date = Field(..., description="Дата заключения договора и первой выдачи")
    construction_deadline: date = Field(..., description="Нормативный срок сдачи дома и регистрации залога")
    days_overdue: int = Field(default=0, ge=0, description="Дни просроченной задолженности (ПЗ)")
    projected_wall_material: WallMaterial = Field(
        default=WallMaterial.BRICK,
        description="Проектный материал несущих стен по смете"
    )
    projected_floors: int = Field(default=1, ge=1, le=5, description="Проектная этажность здания")
    planned_area_sqm: float = Field(default=120.0, gt=0, description="Проектная площадь дома (кв. м)")
    land_category_vri: str = Field(
        default="Для индивидуального жилищного строительства (ИЖС)",
        description="Вид разрешенного использования (ВРИ) земельного участка"
    )
    address: str = Field(default="Новосибирская область", description="Адрес или описание местоположения")
    centroid_lat: Optional[float] = Field(default=None, description="Широта центра участка WGS84")
    centroid_lon: Optional[float] = Field(default=None, description="Долгота центра участка WGS84")
    construction_start_date: str = Field(default="Нет", description="Дата начала строительства")
    planned_completion_date_str: str = Field(default="Нет", description="Плановая дата")
    actual_completion_date_str: str = Field(default="Нет", description="Фактическая дата")
    has_pz: str = Field(default="Нет", description="Наличие ПЗ")
    is_registered: str = Field(default="Нет", description="ОН зарегистрирован")
    is_pledged: str = Field(default="Нет", description="Залог")
    has_aerial_photo: str = Field(default="Да", description="Наличие фото с неба")
    is_minimal_schema: bool = Field(default=False, description="Строгий формат реестра заказчика без додуманных полей")

    @property
    def clean_cadastral(self) -> str:
        """Cadastral number stripped of colons for filesystem and dataset matching."""
        return self.cadastral_number.replace(":", "").replace(" ", "")

    @field_validator("cadastral_number")
    @classmethod
    def validate_cadastral(cls, v: str) -> str:
        v = v.strip()
        digits = v.replace(":", "").replace(" ", "")
        if not digits.isdigit() or len(digits) < 10:
            raise ValueError(f"Некорректный формат кадастрового номера: {v}")
        return v
