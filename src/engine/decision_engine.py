"""Deterministic Banking Decision Engine for AgentDrone.

Implements the official 5-strategy decision matrix of Siberian Bank (Sber):
1. Green Corridor / Tranche Approval - Automated STP approval
2. Financial Restructuring - Loan holiday / schedule prolongation
3. Collateral Pledge Registration - 100% completed building registration
4. Security Audit / Fraud Alert - Material substitution / commercial use
5. Legal Enforcement / Recovery - Imminent deadline default or 90+ days overdue

Supports both:
- Direct CV inspection analysis (used by Streamlit UI and automated pipelines)
- Full banking domain model analysis (used by credit registry, reports, and audit services)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from ..models.credit import CreditObject as BankCreditObject
from ..models.enums import BankStrategy, RiskLevel, StageCategory, WallMaterial as BankWallMaterial
from ..models.inspection import InspectionResult
from ..models.schemas import (
    CreditObject as CVCreditObject,
    CVInspectionResult,
    StrategyRecommendation as CVStrategyRecommendation,
    StrategyType,
)
from ..models.strategy import StrategyRecommendation as BankStrategyRecommendation


class DecisionEngine:
    """Evaluates credit and inspection data against Siberian Bank risk rules."""

    def __init__(self, reference_date: Optional[date] = None):
        self.reference_date = reference_date

    @classmethod
    def evaluate(
        cls,
        *args,
        **kwargs,
    ) -> Any:
        """Universal evaluate entrypoint supporting both CV-based and Domain-based workflows."""
        # Handle instance or class call
        instance: Optional[DecisionEngine] = None
        remaining_args = list(args)
        if remaining_args and isinstance(remaining_args[0], DecisionEngine):
            instance = remaining_args.pop(0)

        # 1. Determine if this is a CV-based evaluation (from UI / cv pipeline)
        cv_arg: Optional[CVInspectionResult] = None
        credit_arg: Optional[CVCreditObject] = None
        for a in remaining_args:
            if isinstance(a, CVInspectionResult):
                cv_arg = a
            elif isinstance(a, CVCreditObject):
                credit_arg = a

        if cv_arg is not None or "cv_result" in kwargs:
            res_cv = cv_arg or kwargs["cv_result"]
            res_cred = credit_arg or kwargs.get("credit_obj") or kwargs.get("credit_object")
            return cls.evaluate_cv(res_cv, res_cred)

        # 2. Domain-based evaluation (CreditObject + InspectionResult)
        credit = remaining_args[0] if remaining_args else kwargs.get("credit")
        inspection = remaining_args[1] if len(remaining_args) > 1 else kwargs.get("inspection")
        eval_date = remaining_args[2] if len(remaining_args) > 2 else kwargs.get("eval_date")
        ref_date = instance.reference_date if instance else None
        return cls.evaluate_inspection(credit, inspection, eval_date=eval_date, reference_date=ref_date)

    # -------------------------------------------------------------------------
    # CV & UI Evaluation Pipeline
    # -------------------------------------------------------------------------
    @staticmethod
    def evaluate_cv(
        cv_result: CVInspectionResult,
        credit_obj: Optional[CVCreditObject] = None,
    ) -> CVStrategyRecommendation:
        """Evaluate drone CV inspection against credit agreement covenants."""
        if credit_obj is None:
            from ..ingestion.credit_registry import CreditRegistry
            credit_obj = CreditRegistry.get_credit_object(cv_result.cadastral_id)
        if credit_obj is None:
            credit_obj = CVCreditObject(cadastral_id=cv_result.cadastral_id)
        reasons: list[str] = []

        # Determine effective overdue days by comparing inspection flight date to planned completion date
        def _parse_date_flexible(val: Any) -> Optional[date]:
            if not val:
                return None
            if isinstance(val, date):
                return val
            s = str(val).strip()
            if s.lower() in ("нет", "н/д", "none", "", "не указан"):
                return None
            s_clean = s.split()[0]
            for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y_%m_%d", "%d/%m/%Y"):
                try:
                    return datetime.strptime(s_clean, fmt).date()
                except ValueError:
                    pass
            return None

        insp_date: Optional[date] = None
        if getattr(cv_result, "inspection_date", None):
            insp_date = _parse_date_flexible(cv_result.inspection_date)
        if not insp_date:
            for sm in getattr(cv_result, "shots_metadata", []):
                if getattr(sm, "timestamp", None):
                    insp_date = _parse_date_flexible(sm.timestamp)
                    if insp_date:
                        break

        plan_date: Optional[date] = _parse_date_flexible(getattr(credit_obj, "planned_completion_date", None))
        if not plan_date and hasattr(credit_obj, "cadastral_id"):
            try:
                from ..engine.registry import CreditRegistry as JsonCreditRegistry
                j_reg = JsonCreditRegistry()
                j_obj = j_reg.get_by_cadastral(credit_obj.cadastral_id)
                if j_obj and j_obj.construction_deadline:
                    plan_date = j_obj.construction_deadline
            except Exception:
                pass

        flight_days_overdue = 0
        if insp_date and plan_date:
            if insp_date > plan_date:
                flight_days_overdue = (insp_date - plan_date).days

        base_days_overdue = int(getattr(credit_obj, "days_overdue", 0))
        effective_days_overdue = max(base_days_overdue, flight_days_overdue)

        # 1. STOP-FRAUD CHECK (Highest Priority)
        if cv_result.fraud_risk_flag or len(cv_result.fraud_reasons) > 0:
            reasons.extend(cv_result.fraud_reasons)
            if not cv_result.geocheck.is_correct_plot:
                reasons.append(
                    f"Нарушение геопривязки: отклонение от кадастрового участка "
                    f"{cv_result.geocheck.distance_to_cadastre_m:.1f}м (>75м)"
                )
            return CVStrategyRecommendation(
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
        if effective_days_overdue > 90 and cv_result.readiness_pct < 25.0:
            reasons.append(
                f"Критическая просрочка вехи строительства: {effective_days_overdue} дней"
            )
            reasons.append(
                f"Фактическая готовность всего {cv_result.readiness_pct:.1f}% при плане сдачи {credit_obj.planned_completion_date}"
            )
            if cv_result.defects_detected:
                reasons.extend(cv_result.defects_detected)

            return CVStrategyRecommendation(
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

        # 3. FAIR OVERDUE EVALUATION FOR UNREGISTERED BUILDINGS (Deadline severely breached)
        is_mortgage_registered = (
            getattr(credit_obj, "is_registered", "Нет") == "Да"
            and getattr(credit_obj, "is_pledged", "Нет") == "Да"
        )
        if effective_days_overdue > 90 and not is_mortgage_registered:
            reasons.append(
                f"Срыв нормативного срока строительства: фактическая просрочка на дату съемки составляет {effective_days_overdue} дн. (план ввода: {credit_obj.planned_completion_date})"
            )
            reasons.append(
                "Постоянный залог (ипотека завершенного жилого дома) в установленный кредитным договором срок не оформлен в ЕГРН"
            )
            reasons.append(
                f"Текущая готовность конструктива {cv_result.readiness_pct:.1f}% ({cv_result.stage_title}), зафиксирован риск долгостроя"
            )
            return CVStrategyRecommendation(
                strategy=StrategyType.RESTRUCTURE_EXTEND,
                strategy_code="STRAT-02",
                title="Реструктуризация / Продление срока этапа и урегулирование просрочки ввода",
                risk_level="MEDIUM",
                action_required="Вызвать заемщика на комиссию по финансовому оздоровлению. Потребовать предоставления документов о вводе в эксплуатацию и регистрации ипотеки. Согласовать график устранения просрочки.",
                reasoning=reasons,
                financial_impact_rub=credit_obj.credit_amount_rub,
                legal_basis="Внутренний регламент мониторинга строительства Сибирского банка Сбера, ст. 811 ГК РФ",
                kalita_status="MANUAL_REVIEW",
            )

        # 4. 100% COMPLETION & COLLATERAL REGISTRATION (Within normal timeline)
        if cv_result.readiness_pct >= 90.0 or cv_result.readiness_stage == 100:
            reasons.append(
                f"Объект завершен строительством: готовность {cv_result.readiness_pct:.1f}% (Стадия 100%)"
            )
            reasons.append("Тепловой контур замкнут, кровля и оконные блоки смонтированы")
            reasons.append("Геопозиция строго соответствует границам кадастрового участка")

            return CVStrategyRecommendation(
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

        # 5. RESTRUCTURING / TIMELINE EXTENSION (Moderate delay or lagging)
        min_required_readiness = 40.0 if credit_obj.tranche_number >= 2 else 0.0
        is_lagging = cv_result.readiness_pct < min_required_readiness
        has_moderate_overdue = 0 < effective_days_overdue <= 90

        if has_moderate_overdue or is_lagging:
            if has_moderate_overdue:
                reasons.append(f"Имеется допустимое отставание от графика: {effective_days_overdue} дн.")
            if is_lagging:
                reasons.append(
                    f"Текущая готовность {cv_result.readiness_pct:.1f}% ниже порога транша №{credit_obj.tranche_number} ({min_required_readiness:.0f}%)"
                )
            reasons.append("Строительные работы ведутся добросовестно, признаков мошенничества не обнаружено")

            return CVStrategyRecommendation(
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

        # 6. APPROVE NEXT TRANCHE (Standard positive scenario)
        reasons.append(
            f"Готовность объекта {cv_result.readiness_pct:.1f}% (Стадия {cv_result.readiness_stage}%) соответствует графику"
        )
        reasons.append(f"Материал стен ({cv_result.detected_material.value}) соответствует заявленному в договоре")
        reasons.append(f"Геопривязка подтверждена, отклонение центроида {cv_result.geocheck.distance_to_cadastre_m:.1f}м")

        return CVStrategyRecommendation(
            strategy=StrategyType.APPROVE_TRANCHE,
            strategy_code="STRAT-01",
            title=f"Автоматическое одобрение выдачи транша №{credit_obj.tranche_number} (STP Сбербанк)",
            risk_level="LOW",
            action_required=f"Акцептовать транш №{credit_obj.tranche_number} на сумму {credit_obj.credit_amount_rub * 0.35:,.0f} руб. в системе мониторинга залогов.",
            reasoning=reasons,
            financial_impact_rub=credit_obj.credit_amount_rub * 0.35,
            legal_basis="Кредитный договор, п. 3.2 'Условия выдачи траншей'",
            kalita_status="APPROVED_AUTO",
        )

    # -------------------------------------------------------------------------
    # Banking Covenants & Inspection Domain Evaluation
    # -------------------------------------------------------------------------
    @staticmethod
    def evaluate_inspection(
        credit: BankCreditObject,
        inspection: InspectionResult,
        eval_date: Optional[date] = None,
        reference_date: Optional[date] = None,
    ) -> BankStrategyRecommendation:
        """Run hierarchical rule engine to determine bank strategy."""
        current_date = eval_date or reference_date or inspection.inspection_date
        days_to_deadline = (credit.construction_deadline - current_date).days

        # 1. STOP-FRAUD: Security Audit & Non-Target Use Check
        if inspection.is_commercial_building_risk:
            return BankStrategyRecommendation(
                strategy=BankStrategy.FRAUD_ALERT,
                decision_code="FRAUD_COMMERCIAL_USE",
                risk_level=RiskLevel.CRITICAL,
                confidence=0.96,
                is_automated=False,
                reasons=[
                    f"Выявлены признаки нецелевого использования земельного участка {credit.cadastral_number}",
                    "На участке ИЖС возведен коммерческий объект (магазин / склад / сервис)",
                    f"ВРИ участка: '{credit.land_category_vri}' не допускает коммерческую эксплуатацию",
                ],
                action_items=[
                    "Немедленно приостановить выдачу кредитных траншей",
                    "Направить материалы в Управление безопасности Сибирского банка",
                    "Инициировать очный комиссионный выезд инспектора залоговой службы",
                    "Подготовить уведомление заемщику о нецелевом использовании средств",
                ],
                summary=(
                    f"КРИТИЧЕСКИЙ РИСК (ФРОД): На участке {credit.cadastral_number} зафиксирован "
                    "коммерческий объект вместо индивидуального жилого дома. Финансирование заблокировано."
                ),
            )

        # Check for material substitution (cheap frame/shield instead of brick/block)
        if not inspection.material_match:
            is_critical_downgrade = (
                credit.projected_wall_material in (BankWallMaterial.BRICK, BankWallMaterial.CONCRETE_BLOCK)
                and inspection.detected_wall_material == BankWallMaterial.FRAME_PANEL
            )
            return BankStrategyRecommendation(
                strategy=BankStrategy.FRAUD_ALERT,
                decision_code="FRAUD_MATERIAL_SUBSTITUTION",
                risk_level=RiskLevel.HIGH if is_critical_downgrade else RiskLevel.MEDIUM,
                confidence=0.92,
                is_automated=False,
                reasons=[
                    f"Несоответствие фактического материала стен проектно-сметной документации на участке {credit.cadastral_number}",
                    f"По смете заявлен: '{credit.projected_wall_material.value}'",
                    f"По данным аэросъемки обнаружен: '{inspection.detected_wall_material.value}'",
                    "Удешевление несущего конструктива снижает ликвидационную стоимость залога",
                ],
                action_items=[
                    "Заблокировать выдачу следующего транша до выяснения обстоятельств",
                    "Запросить у заемщика исполнительную документацию и сертификаты на материалы",
                    "Провести переоценку залоговой стоимости с дисконтом",
                ],
                summary=(
                    f"РИСК ПОДМЕНЫ МАТЕРИАЛОВ: Заемщик использует '{inspection.detected_wall_material.value}' "
                    f"вместо проектного '{credit.projected_wall_material.value}'. Требуется экспертиза."
                ),
            )

        # Boundary violation check (risk of demolition order)
        if inspection.boundary_violation:
            return BankStrategyRecommendation(
                strategy=BankStrategy.FRAUD_ALERT,
                decision_code="FRAUD_BOUNDARY_VIOLATION",
                risk_level=RiskLevel.HIGH,
                confidence=0.90,
                is_automated=False,
                reasons=[
                    f"Нарушен минимальный отступ от границ участка (<3 метров) на участке {credit.cadastral_number}",
                    f"Фактическое расстояние: {inspection.distance_to_boundary_m or 0.0:.1f} м (норматив >= 3.0 м)",
                    "Риск признания строения самовольной постройкой по ст. 222 ГК РФ с угрозой сноса",
                ],
                action_items=[
                    "Приостановить выдачу средств",
                    "Направить запрос в Росреестр и архитектурный отдел муниципалитета",
                    "Потребовать градостроительный план земельного участка (ГПЗУ)",
                ],
                summary=(
                    f"ЮРИДИЧЕСКИЙ РИСК: Нарушен нормативный отступ от межи ({inspection.distance_to_boundary_m or 0.0:.1f}м). "
                    "Угроза иска о сносе самовольной постройки по ст. 222 ГК РФ."
                ),
            )

        # 2. LEGAL RECOVERY / FORECLOSURE: Imminent Default or Deep Overdue
        is_deep_overdue = credit.days_overdue >= 90
        is_deadline_expired_no_progress = days_to_deadline < 90 and inspection.physical_readiness_pct < 20.0

        if is_deep_overdue or is_deadline_expired_no_progress:
            reasons = []
            if is_deep_overdue:
                reasons.append(f"Длительная просроченная задолженность: {credit.days_overdue} дней (порог дефолта >=90 дней)")
            if is_deadline_expired_no_progress:
                reasons.append(
                    f"До нормативного срока сдачи ({credit.construction_deadline}) осталось {days_to_deadline} дней, "
                    f"а готовность объекта составляет всего {inspection.physical_readiness_pct:.1f}%"
                )
            if inspection.is_construction_abandoned:
                reasons.append("Объект имеет признаки заброшенности и не законсервирован")

            return BankStrategyRecommendation(
                strategy=BankStrategy.LEGAL_RECOVERY,
                decision_code="LEGAL_DEEP_OVERDUE_90_PLUS" if is_deep_overdue else "LEGAL_IMMINENT_DEADLINE_DEFAULT",
                risk_level=RiskLevel.CRITICAL,
                confidence=0.97,
                is_automated=False,
                reasons=reasons,
                action_items=[
                    "Передать кредитное досье в Юридическое управление и УРПА",
                    "Подготовить требование о досрочном возврате всей суммы кредита",
                    "Наложить арест на земельный участок и зарегистрировать ограничение в ЕГРН",
                    "Инициировать судебное взыскание и подготовку к реализации на торгах",
                ],
                summary=(
                    f"ДЕФОЛТ / ВЗЫСКАНИЕ: Кредит классифицирован как проблемный (ПЗ={credit.days_overdue} дн., "
                    f"готовность {inspection.physical_readiness_pct:.0f}%). Инициировать судебное обращение взыскания на залог."
                ),
            )

        # 3. PLEDGE REGISTRATION: Completed Building (100% Ready)
        if inspection.stage == StageCategory.STAGE_100 or inspection.physical_readiness_pct >= 85.0:
            return BankStrategyRecommendation(
                strategy=BankStrategy.PLEDGE_REGISTRATION,
                decision_code="PLEDGE_REGISTRATION_READY",
                risk_level=RiskLevel.LOW,
                confidence=0.99,
                is_automated=True,
                reasons=[
                    f"Строительство жилого дома на участке {credit.cadastral_number} завершено",
                    f"Физическая готовность объекта составляет {inspection.physical_readiness_pct:.1f}% (Стадия 100%)",
                    "Конструктивные элементы смонтированы в полном объеме (фундамент, стены, кровля, контур)",
                    "Фактический материал соответствует проектно-сметной документации",
                ],
                action_items=[
                    "Направить заемщику уведомление о необходимости оформления права собственности",
                    "Зарегистрировать ипотеку готового жилого дома в пользу ПАО Сбербанк в ЕГРН",
                    "Провести оценку рыночной стоимости завершенного объекта",
                    "Снизить процентную ставку по кредиту согласно условиям госпрограммы ИЖС",
                ],
                summary=(
                    f"РЕГИСТРАЦИЯ ЗАЛОГА: Объект готов ({inspection.physical_readiness_pct:.0f}%). "
                    "Необходимо зарегистрировать ипотеку в Росреестре и закрыть этап стройки."
                ),
            )

        # 4. RESTRUCTURING: Financial Recovery / Delayed Schedule
        is_significantly_behind = (
            days_to_deadline < 180 and inspection.stage == StageCategory.STAGE_0
        ) or (
            days_to_deadline < 90 and inspection.physical_readiness_pct < 50.0
        )
        has_active_overdue = 0 < credit.days_overdue < 90

        if has_active_overdue or is_significantly_behind:
            reasons = []
            if has_active_overdue:
                reasons.append(f"Имеется текущая просроченная задолженность ({credit.days_overdue} дней, до 90 дней)")
            if is_significantly_behind:
                reasons.append(
                    f"Отставание от графика: до сдачи {days_to_deadline} дн., текущая готовность {inspection.physical_readiness_pct:.1f}%"
                )
            reasons.append("Заемщик идет на контакт, целевое назначение участка соблюдается")

            return BankStrategyRecommendation(
                strategy=BankStrategy.RESTRUCTURING,
                decision_code="RESTRUCTURING_BEHIND_SCHEDULE" if is_significantly_behind else "RESTRUCTURING_OVERDUE_WARNING",
                risk_level=RiskLevel.MEDIUM if not has_active_overdue else RiskLevel.HIGH,
                confidence=0.91,
                is_automated=False,
                reasons=reasons,
                action_items=[
                    "Предложить заемщику программу финансового оздоровления (кредитные каникулы до 6 мес.)",
                    "Согласовать продление нормативного срока строительства с сохранением ставки",
                    "Разработать корректирующий график траншей для завершения коробки",
                ],
                summary=(
                    f"ФИНАНСОВОЕ ОЗДОРОВЛЕНИЕ: Зафиксировано отставание / просрочка ({credit.days_overdue} дн.). "
                    "Рекомендуется реструктуризация для спасения кредита до наступления дефолта."
                ),
            )

        # 5. GREEN CORRIDOR: Tranche Approval (STP)
        return BankStrategyRecommendation(
            strategy=BankStrategy.TRANCHE_APPROVAL,
            decision_code="TRANCHE_APPROVAL_STP",
            risk_level=RiskLevel.LOW,
            confidence=0.98,
            is_automated=True,
            reasons=[
                "Строительство ведется строго в соответствии с графиком кредитного договора",
                f"Фактическая готовность {inspection.physical_readiness_pct:.1f}% подтверждает целевой этап",
                f"Фактический материал ({inspection.detected_wall_material.value}) соответствует смете",
                "Просроченная задолженность отсутствует (ПЗ = 0 дней)",
                "Градостроительные нормы и отступы соблюдены",
            ],
            action_items=[
                (
                    f"Автоматически согласовать перевод очередного транша №{credit.tranche_number} "
                    f"на сумму {credit.next_tranche_amount:,.2f} руб."
                ),
                "Обновить статус объекта в системе мониторинга залогов на 'Зеленый коридор'",
                "Запланировать следующий контрольный облет БПЛА через 60 дней",
            ],
            summary=(
                f"ЗЕЛЕНЫЙ КОРИДОР: Стройка в графике ({inspection.physical_readiness_pct:.0f}%), нарушений нет. "
                f"Автоматическое согласование транша №{credit.tranche_number} ({credit.next_tranche_amount:,.0f} руб.)."
            ),
        )
