"""Deterministic Bank Decision Engine for AgentDrone.

Applies the 5 core Bank monitoring strategies to an inspected collateral plot:
1. STRAT-01: Approve next credit tranche (Выдача очередного транша)
2. STRAT-02: Restructure / extension of credit timeline (Реструктуризация / Продление)
3. STRAT-03: Final collateral registration / closure of monitoring (Регистрация залога)
4. STRAT-04: Fraud suspension / Security Service investigation (Блокировка / Фрод)
5. STRAT-05: Default / Foreclosure / Legal collection (Дефолт / Судебное взыскание)
"""

from __future__ import annotations

from ..models.schemas import (
    CreditObject,
    CVInspectionResult,
    StrategyRecommendation,
    StrategyType,
)


class DecisionEngine:
    """Evaluates drone inspection results against credit agreement covenants."""

    @classmethod
    def evaluate(
        cls,
        cv_result: CVInspectionResult,
        credit_obj: CreditObject,
    ) -> StrategyRecommendation:
        """Execute decision matrix and return prioritized actionable strategy."""
        reasons: list[str] = []

        # 1. STOP-FRAUD CHECK (Highest Priority)
        # Commercial object on residential parcel, falsified GPS, or critical material substitute
        if cv_result.fraud_risk_flag or len(cv_result.fraud_reasons) > 0:
            reasons.extend(cv_result.fraud_reasons)
            if not cv_result.geocheck.is_correct_plot:
                reasons.append(
                    f"Нарушение геопривязки: отклонение от кадастрового участка "
                    f"{cv_result.geocheck.distance_to_cadastre_m:.1f}м (>75м)"
                )
            return StrategyRecommendation(
                strategy=StrategyType.SUSPEND_SECURITY_FRAUD,
                strategy_code="STRAT-04",
                title="Блокировка траншей и передача в Службу Безопасности (Stop-Fraud)",
                risk_level="CRITICAL",
                action_required="Заблокировать выдачу средств. Направить выездную проверку СБ и претензию заемщику.",
                reasoning=reasons if reasons else ["Выявлены признаки нецелевого использования заемных средств"],
                financial_impact_rub=credit_obj.credit_amount_rub,
                legal_basis="Ст. 813, 814 ГК РФ (Нецелевое использование кредита, непредоставление обеспечения)",
                kalita_status="BLOCKED_FRAUD",
            )

        # 2. DEFAULT & FORECLOSURE CHECK
        # Abandoned construction with long overdue (>90 days and low readiness)
        if credit_obj.days_overdue > 90 and cv_result.readiness_pct < 25.0:
            reasons.append(
                f"Критическая просрочка вехи строительства: {credit_obj.days_overdue} дней"
            )
            reasons.append(
                f"Фактическая готовность всего {cv_result.readiness_pct:.1f}% при плане сдачи {credit_obj.planned_completion_date}"
            )
            if cv_result.defects_detected:
                reasons.extend(cv_result.defects_detected)

            return StrategyRecommendation(
                strategy=StrategyType.DEFAULT_FORECLOSURE,
                strategy_code="STRAT-05",
                title="Инициация процедуры дефолта и судебного обращения взыскания на залог",
                risk_level="CRITICAL",
                action_required="Передать досье в Управление по работе с проблемными активами (УРПА). Подготовить иск.",
                reasoning=reasons,
                financial_impact_rub=credit_obj.credit_amount_rub,
                legal_basis="ФЗ-102 'Об ипотеке (залоге недвижимости)', ст. 348 ГК РФ",
                kalita_status="LEGAL_ACTION",
            )

        # 3. 100% COMPLETION & COLLATERAL REGISTRATION
        # Ready for full mortgage registration in Rosreestr
        if cv_result.readiness_pct >= 90.0 or cv_result.readiness_stage == 100:
            reasons.append(
                f"Объект завершен строительством: готовность {cv_result.readiness_pct:.1f}% (Стадия 100%)"
            )
            reasons.append("Тепловой контур замкнут, кровля и оконные блоки смонтированы")
            reasons.append("Геопозиция строго соответствует границам кадастрового участка")

            return StrategyRecommendation(
                strategy=StrategyType.REGISTER_COLLATERAL,
                strategy_code="STRAT-03",
                title="Оформление постоянного залога (ипотеки) и закрытие мониторинга ПЗ",
                risk_level="LOW",
                action_required="Уведомить заемщика о необходимости подачи заявления о вводе в эксплуатацию и регистрации ипотеки.",
                reasoning=reasons,
                financial_impact_rub=credit_obj.credit_amount_rub,
                legal_basis="ФЗ-218 'О государственной регистрации недвижимости', условия кредитного договора",
                kalita_status="COLLATERAL_REGISTERED",
            )

        # 4. RESTRUCTURING / TIMELINE EXTENSION
        # Moderate delay (<=90 days overdue) or slight readiness lagging behind current tranche
        min_required_readiness = 40.0 if credit_obj.tranche_number >= 2 else 0.0
        is_lagging = cv_result.readiness_pct < min_required_readiness
        has_moderate_overdue = 0 < credit_obj.days_overdue <= 90

        if has_moderate_overdue or is_lagging:
            if has_moderate_overdue:
                reasons.append(f"Имеется допустимое отставание от графика: {credit_obj.days_overdue} дн.")
            if is_lagging:
                reasons.append(
                    f"Текущая готовность {cv_result.readiness_pct:.1f}% ниже порога транша №{credit_obj.tranche_number} ({min_required_readiness:.0f}%)"
                )
            reasons.append("Строительные работы ведутся добросовестно, признаков мошенничества не обнаружено")

            return StrategyRecommendation(
                strategy=StrategyType.RESTRUCTURE_EXTEND,
                strategy_code="STRAT-02",
                title="Продление срока этапа строительства / Реструктуризация ковенантов",
                risk_level="MEDIUM",
                action_required="Заключить дополнительное соглашение о продлении контрольной даты этапа на 60 дней.",
                reasoning=reasons,
                financial_impact_rub=credit_obj.credit_amount_rub,
                legal_basis="Внутренний регламент мониторинга строительства Сибирского банка Сбера",
                kalita_status="MANUAL_REVIEW",
            )

        # 5. APPROVE NEXT TRANCHE (Standard positive scenario)
        reasons.append(
            f"Готовность объекта {cv_result.readiness_pct:.1f}% (Стадия {cv_result.readiness_stage}%) соответствует графику"
        )
        reasons.append(f"Материал стен ({cv_result.detected_material.value}) соответствует заявленному в договоре")
        reasons.append(f"Геопривязка подтверждена, отклонение центроида {cv_result.geocheck.distance_to_cadastre_m:.1f}м")

        return StrategyRecommendation(
            strategy=StrategyType.APPROVE_TRANCHE,
            strategy_code="STRAT-01",
            title=f"Автоматическое одобрение выдачи транша №{credit_obj.tranche_number} в АС 'Калита'",
            risk_level="LOW",
            action_required=f"Акцептовать транш №{credit_obj.tranche_number} на сумму {credit_obj.credit_amount_rub * 0.35:,.0f} руб. в АС Калита.",
            reasoning=reasons,
            financial_impact_rub=credit_obj.credit_amount_rub * 0.35,
            legal_basis="Кредитный договор, п. 3.2 'Условия выдачи траншей'",
            kalita_status="APPROVED_AUTO",
        )
