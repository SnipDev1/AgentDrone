"""Prompt templates for GigaChat API inspector summary generation.

Formats structured credit and inspection telemetry into prompt payloads
for the LLM to generate official summaries for AS 'Kalita'.
"""

import json
from ..models import CreditObject, InspectionResult, StrategyRecommendation

SYSTEM_INSPECTOR_PROMPT = (
    "Вы — главный эксперт Управления мониторинга обеспечения и проблемных активов "
    "Сибирского банка ПАО Сбербанк.\n"
    "Ваша задача — составить официальное, юридически и технически выверенное Экспертное резюме "
    "инспектора для внесения в АС «Калита» по результатам обследования объекта индивидуального "
    "жилищного строительства (ИЖС) с использованием БПЛА.\n\n"
    "Требования к резюме:\n"
    "1. Деловой, строгий тон банковского отчета.\n"
    "2. Четкая фиксация: статус строительной готовности (0% / 50% / 100%), соответствие проекту "
    "по смете, соблюдение сроков.\n"
    "3. Оценка залогового риска и целевого использования участка (ИЖС vs коммерция, подмена материалов).\n"
    "4. Однозначное утверждение рекомендованной стратегии Банка и предписания ответственным службам.\n"
    "5. Объем: 2-3 емких, информативных абзаца.\n"
)


def build_inspector_prompt(
    credit: CreditObject,
    inspection: InspectionResult,
    recommendation: StrategyRecommendation
) -> str:
    """Build structured user prompt with all domain variables."""
    payload = {
        "кредитный_договор": {
            "номер": credit.credit_id,
            "заемщик": credit.borrower_name,
            "кадастровый_номер": credit.cadastral_number,
            "сумма_кредита_руб": credit.credit_amount,
            "выдано_руб": credit.current_disbursed,
            "текущий_транш": credit.tranche_number,
            "просрочка_дней": credit.days_overdue,
            "дедлайн_стройки": str(credit.construction_deadline),
            "проектный_материал": credit.projected_wall_material.value,
        },
        "результаты_бпла_осмотра": {
            "дата_съемки": str(inspection.inspection_date),
            "физическая_готовность_процент": inspection.physical_readiness_pct,
            "нормативная_стадия": inspection.stage.value,
            "фактический_материал": inspection.detected_wall_material.value,
            "соответствие_материала": inspection.material_match,
            "отступ_до_границы_м": inspection.distance_to_boundary_m,
            "нарушение_границ": inspection.boundary_violation,
            "риск_коммерческого_использования": inspection.is_commercial_building_risk,
            "дефекты": [d.model_dump() for d in inspection.defects],
        },
        "решение_decision_engine": {
            "стратегия": recommendation.strategy.value,
            "код_решения": recommendation.decision_code,
            "уровень_риска": recommendation.risk_level.value,
            "ключевые_аргументы": recommendation.reasons,
            "предписанные_действия": recommendation.action_items,
        }
    }

    return (
        "Сформируйте официальное Экспертное резюме инспектора для АС «Калита» "
        f"на основе следующих проверенных данных мониторинга:\n\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
