"""Prompt templates for GigaChat API inspector summary generation.

Formats structured credit and inspection telemetry into prompt payloads
for the LLM to generate official summaries for Sberbank collateral monitoring.
Supports both Banking Domain models and CV Inspection pipeline schemas.
"""

from __future__ import annotations

import json
from typing import Any

SYSTEM_INSPECTOR_PROMPT = (
    "Вы — главный эксперт Управления мониторинга обеспечения и проблемных активов "
    "Сибирского банка ПАО Сбербанк.\n"
    "Ваша задача — составить официальное, юридически и технически выверенное Экспертное резюме "
    "инспектора для внесения в кредитное досье залоговой службы Сбербанка по результатам обследования объекта "
    "индивидуального жилищного строительства (ИЖС) с использованием БПЛА.\n\n"
    "Требования к резюме:\n"
    "1. Деловой, строгий тон банковского отчета.\n"
    "2. Четкая фиксация: статус строительной готовности (0% / 50% / 100%), соответствие проекту "
    "по смете, соблюдение сроков.\n"
    "3. Оценка залогового риска и целевого использования участка (ИЖС vs коммерция, подмена материалов).\n"
    "4. Однозначное утверждение рекомендованной стратегии Банка и предписания ответственным службам.\n"
    "5. Объем: 2-3 емких, информативных абзаца.\n"
)


def extract_context_payload(
    credit: Any,
    inspection: Any,
    recommendation: Any,
) -> dict[str, Any]:
    """Extract unified dictionary payload regardless of model variant."""
    # 1. Credit details
    credit_id = getattr(credit, "credit_id", None) or getattr(credit, "credit_agreement_number", "CR-2026")
    borrower = getattr(credit, "borrower_name", "Заемщик")
    cadastral = (
        getattr(credit, "cadastral_number", None)
        or getattr(credit, "cadastral_id", None)
        or getattr(inspection, "cadastral_id", "54:00:000000:0")
    )
    credit_amount = getattr(credit, "credit_amount", None) or getattr(credit, "credit_amount_rub", 0.0)
    disbursed = getattr(credit, "current_disbursed", None)
    if disbursed is None:
        tranche_num = getattr(credit, "tranche_number", 1)
        disbursed = (credit_amount / 3.0) * tranche_num
    days_overdue = getattr(credit, "days_overdue", 0)
    deadline = getattr(credit, "construction_deadline", None) or getattr(credit, "planned_completion_date", "2027-01-01")

    proj_mat = getattr(credit, "projected_wall_material", None)
    if proj_mat and hasattr(proj_mat, "value"):
        proj_mat_val = proj_mat.value
    else:
        proj_mat_val = str(getattr(credit, "declared_wall_material", "") or "")

    # 2. Inspection details
    insp_date = str(getattr(inspection, "inspection_date", "2026-09-24"))
    readiness_pct = getattr(inspection, "physical_readiness_pct", None)
    if readiness_pct is None:
        readiness_pct = getattr(inspection, "readiness_pct", 0.0)

    stage_val = getattr(inspection, "stage", None)
    if stage_val and hasattr(stage_val, "value"):
        stage_str = stage_val.value
    else:
        stage_stage = getattr(inspection, "readiness_stage", 50)
        stage_title = getattr(inspection, "stage_title", f"Стадия {stage_stage}%")
        stage_str = f"{stage_stage}% ({stage_title})"

    det_mat = getattr(inspection, "detected_wall_material", None) or getattr(inspection, "detected_material", None)
    det_mat_val = det_mat.value if hasattr(det_mat, "value") else str(det_mat or "Не определен")

    mat_match = getattr(inspection, "material_match", None)
    if mat_match is None:
        if not proj_mat_val:
            mat_match = True
        else:
            mat_match = proj_mat_val.lower() in det_mat_val.lower() or det_mat_val.lower() in proj_mat_val.lower()

    dist_boundary = getattr(inspection, "distance_to_boundary_m", None)
    bound_violation = getattr(inspection, "boundary_violation", False)

    geocheck = getattr(inspection, "geocheck", None)
    if geocheck:
        geo_valid = getattr(geocheck, "is_correct_plot", True)
        geo_dist = getattr(geocheck, "distance_to_cadastre_m", 0.0)
    else:
        geo_valid = getattr(inspection, "geo_match", True)
        geo_dist = 0.0

    is_commercial = getattr(inspection, "is_commercial_building_risk", False) or getattr(inspection, "fraud_risk_flag", False)
    defects = getattr(inspection, "defects_detected", []) or [
        d.model_dump() if hasattr(d, "model_dump") else str(d) for d in getattr(inspection, "defects", [])
    ]

    # 3. Decision details
    strat_obj = getattr(recommendation, "strategy", None)
    strat_name = strat_obj.value if hasattr(strat_obj, "value") else str(strat_obj or "Выдача транша")
    strat_code = getattr(recommendation, "decision_code", None) or getattr(recommendation, "strategy_code", "STRAT-01")
    risk_level = getattr(recommendation, "risk_level", None)
    risk_level_val = risk_level.value if hasattr(risk_level, "value") else str(risk_level or "LOW")
    reasons = getattr(recommendation, "reasons", None) or getattr(recommendation, "reasoning", [])
    action_items = getattr(recommendation, "action_items", None) or [getattr(recommendation, "action_required", "")]

    return {
        "кредитный_договор": {
            "номер": credit_id,
            "заемщик": borrower,
            "кадастровый_номер": cadastral,
            "сумма_кредита_руб": credit_amount,
            "выдано_руб": disbursed,
            "просрочка_дней": days_overdue,
            "дедлайн_стройки": str(deadline),
            "проектный_материал": proj_mat_val,
        },
        "результаты_бпла_осмотра": {
            "дата_съемки": insp_date,
            "физическая_готовность_процент": readiness_pct,
            "нормативная_стадия": stage_str,
            "фактический_материал": det_mat_val,
            "соответствие_материала": mat_match,
            "геовалидация": {
                "участок_подтвержден": geo_valid,
                "отклонение_координат_м": geo_dist,
            },
            "отступ_до_границы_м": dist_boundary,
            "нарушение_границ": bound_violation,
            "риск_коммерческого_использования": is_commercial,
            "дефекты": defects,
        },
        "решение_decision_engine": {
            "стратегия": strat_name,
            "код_решения": strat_code,
            "уровень_риска": risk_level_val,
            "ключевые_аргументы": reasons,
            "предписанные_действия": action_items,
        },
    }


def build_inspector_prompt(
    credit: Any,
    inspection: Any,
    recommendation: Any,
) -> str:
    """Build structured user prompt with all domain variables."""
    payload = extract_context_payload(credit, inspection, recommendation)
    return (
        "Сформируйте официальное Экспертное резюме инспектора для залоговой службы Сбербанка "
        f"на основе следующих проверенных данных мониторинга:\n\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


SYSTEM_DYNAMICS_PROMPT = (
    "Вы — ведущий эксперт Управления строительного аудита и залогового мониторинга "
    "Сибирского банка ПАО Сбербанк.\n"
    "Ваша задача — подготовить официальное Экспертное заключение ИИ-инспектора по ДИНАМИКЕ СТРОИТЕЛЬСТВА "
    "объекта ИЖС по результатам повторной аэрофотосъемки БПЛА (разновременной мониторинг) "
    "для внесения в кредитное досье залоговой службы Сбербанка.\n\n"
    "Требования к заключению:\n"
    "1. Строгий деловой тон кредитного отчета Сбербанка.\n"
    "2. Анализ динамики: прирост готовности в %, интервал в днях, расчетный темп СМР в месяц (%/мес).\n"
    "3. Детализация фактически возведенных конструкций за отчетный период (нулевой цикл, коробка, кровля, контур).\n"
    "4. Сопоставление фактических темпов с нормативным графиком кредитования и плановым сроком ввода.\n"
    "5. Итоговый вывод для кредитного мониторинга Сбербанка: подтверждение целевого расходования предыдущего транша и "
    "согласование выдачи очередного транша / реструктуризация / проведение очной строительно-технической экспертизы.\n"
    "6. Объем: 2-3 емких, аргументированных абзаца."
)


def build_dynamics_prompt(dynamics: Any, credit: Any = None) -> str:
    """Build structured prompt for AI Agent multi-temporal construction dynamics analysis."""
    snapshots_data = []
    for s in getattr(dynamics, "snapshots", []):
        snapshots_data.append({
            "дата_съемки": s.flight_date,
            "готовность_процент": s.readiness_pct,
            "стадия": s.stage_title,
            "элементы": {
                "фундамент": s.has_foundation,
                "стены": s.has_walls,
                "кровля": s.has_roof,
                "тепловой_контур": s.has_windows_doors,
            },
            "материал": s.detected_material.value if hasattr(s.detected_material, "value") else str(s.detected_material),
            "количество_кадров": s.shots_count,
        })

    payload = {
        "кадастровый_номер": getattr(dynamics, "cadastral_id", "54:00:000000:0"),
        "заемщик": getattr(credit, "borrower_name", "Заемщик") if credit else "Заемщик",
        "адрес_объекта": getattr(credit, "address", "Новосибирская область") if credit else "Новосибирская область",
        "плановый_срок_ввода": getattr(credit, "planned_completion_date", "Не указан") if credit else "Не указан",
        "дата_начала_стройки": getattr(credit, "construction_start_date", "Не указана") if credit else "Не указана",
        "хронология_вылетов_бпла": snapshots_data,
        "метрики_динамики": {
            "прирост_готовности_процент": getattr(dynamics, "delta_readiness_pct", 0.0),
            "дней_между_съемками": getattr(dynamics, "delta_days", 0),
            "темп_смр_в_месяц": getattr(dynamics, "pace_pct_per_month", 0.0),
            "статус_темпа": getattr(dynamics, "pace_status", "В графике"),
            "зафиксированные_изменения": getattr(dynamics, "elements_progress", []),
        },
    }

    return (
        "Составьте официальное Экспертное заключение ИИ-инспектора по динамике строительства "
        "для залоговой службы Сбербанка на основе следующих данных разновременных съемок БПЛА:\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
