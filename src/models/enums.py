"""Enumerations for AgentDrone domain models.

Defines standard banking classifications, construction stages,
materials, and decision strategies.
"""

from enum import Enum


class StageCategory(str, Enum):
    """Discrete normative construction readiness stage used by Sberbank."""
    STAGE_0 = "0% (Старт / Котлован / Ничего нет)"
    STAGE_50 = "50% (Возведена коробка / Фундамент и стены)"
    STAGE_100 = "100% (Завершенное строение / Закрыт тепловой контур)"

    @classmethod
    def from_percentage(cls, pct: float) -> "StageCategory":
        """Map continuous readiness percentage to bank normative category."""
        if pct < 20.0:
            return cls.STAGE_0
        elif pct < 85.0:
            return cls.STAGE_50
        else:
            return cls.STAGE_100


class WallMaterial(str, Enum):
    """Standard structural wall materials."""
    BRICK = "Кирпич / Каменная кладка"
    CONCRETE_BLOCK = "Газобетонные / Пенобетонные блоки"
    TIMBER = "Дерево / Брус / Бревно"
    FRAME_PANEL = "Каркасно-панельные / Щитовые конструкции"
    UNKNOWN = "Не определено"


class BankStrategy(str, Enum):
    """The 5 target bank loan/collateral strategies."""
    TRANCHE_APPROVAL = "Зеленый коридор (Выдача транша)"
    RESTRUCTURING = "Финансовое оздоровление (Реструктуризация)"
    PLEDGE_REGISTRATION = "Регистрация залога"
    FRAUD_ALERT = "Служба безопасности / Аудит (Фрод)"
    LEGAL_RECOVERY = "Судебное взыскание / Торги (Дефолт)"


class RiskLevel(str, Enum):
    """Credit and collateral risk severity level."""
    LOW = "Низкий (Зеленая зона)"
    MEDIUM = "Умеренный (Желтая зона)"
    HIGH = "Высокий (Оранжевая зона)"
    CRITICAL = "Критический (Красная зона)"


class CameraViewType(str, Enum):
    """Drone camera perspective classification."""
    FACADE = "Перспективный фасад (40-50°)"
    NADIR_LOW = "Надир детальный (кровля)"
    NADIR_HIGH = "Надир обзорный (участок и границы)"
    UNKNOWN = "Неопределенный ракурс"


class CollateralCondition(str, Enum):
    """Normative collateral condition for Appendix 1 (Акт проверки залога)."""
    SATISFACTORY = "Удовлетворительное"
    UNSATISFACTORY = "Неудовлетворительное"
