"""Deterministic Banking Decision Engine for AgentDrone.

Implements the official 5-strategy decision matrix of Siberian Bank (Sber):
1. Green Corridor (Tranche Approval) - Automated STP approval
2. Financial Restructuring - Loan holiday / schedule prolongation
3. Collateral Pledge Registration - 100% completed building registration
4. Security Audit (Fraud Alert) - Material substitution / commercial use
5. Legal Enforcement / Recovery - Imminent deadline default or 90+ days overdue
"""

from datetime import date
from typing import List, Optional

from ..models import (
    CreditObject,
    InspectionResult,
    StrategyRecommendation,
    BankStrategy,
    RiskLevel,
    StageCategory,
    WallMaterial,
)


class DecisionEngine:
    """Evaluates credit and inspection data against Siberian Bank risk rules."""

    def __init__(self, reference_date: Optional[date] = None):
        self.reference_date = reference_date

    def evaluate(
        self,
        credit: CreditObject,
        inspection: InspectionResult,
        eval_date: Optional[date] = None
    ) -> StrategyRecommendation:
        """Run hierarchical rule engine to determine bank strategy."""
        current_date = eval_date or self.reference_date or inspection.inspection_date
        days_to_deadline = (credit.construction_deadline - current_date).days

        # -------------------------------------------------------------
        # 1. STOP-FRAUD: Security Audit & Non-Target Use Check
        # -------------------------------------------------------------
        if inspection.is_commercial_building_risk:
            return StrategyRecommendation(
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
                credit.projected_wall_material in (WallMaterial.BRICK, WallMaterial.CONCRETE_BLOCK)
                and inspection.detected_wall_material == WallMaterial.FRAME_PANEL
            )
            return StrategyRecommendation(
                strategy=BankStrategy.FRAUD_ALERT,
                decision_code="FRAUD_MATERIAL_SUBSTITUTION",
                risk_level=RiskLevel.HIGH if is_critical_downgrade else RiskLevel.MEDIUM,
                confidence=0.92,
                is_automated=False,
                reasons=[
                    "Несоответствие фактического материала стен проектно-сметной документации",
                    f"Проект по кредитному договору: {credit.projected_wall_material.value}",
                    f"Фактически определено CV: {inspection.detected_wall_material.value}",
                    "Риск удешевления сметной стоимости предмета будущего залога",
                ],
                action_items=[
                    "Приостановить согласование следующего транша",
                    "Запросить у заемщика исполнительную документацию и сертификаты на материалы",
                    "Провести повторную проверку экспертом-оценщиком",
                ],
                summary=(
                    f"ВНИМАНИЕ (ФРОД/СБ): Зафиксирована подмена материала стен "
                    f"({inspection.detected_wall_material.value} вместо {credit.projected_wall_material.value}). "
                    "Требуется аудит сметы."
                ),
            )

        if inspection.boundary_violation:
            return StrategyRecommendation(
                strategy=BankStrategy.FRAUD_ALERT,
                decision_code="FRAUD_BOUNDARY_VIOLATION",
                risk_level=RiskLevel.HIGH,
                confidence=0.90,
                is_automated=False,
                reasons=[
                    "Нарушение нормативного отступа 3.0 м от границы земельного участка",
                    f"Фактическое расстояние до межи: {inspection.distance_to_boundary_m or 'менее 3.0'} м",
                    "Риск судебного иска соседей и признания постройки самовольной (ст. 222 ГК РФ)",
                ],
                action_items=[
                    "Запросить градостроительный план земельного участка (ГПЗУ)",
                    "Направить уведомление заемщику о градостроительном нарушении",
                ],
                summary=(
                    f"ЮРИДИЧЕСКИЙ РИСК: Нарушен отступ 3 м до межи участка {credit.cadastral_number}. "
                    "Риск предписания о сносе по ст. 222 ГК РФ."
                ),
            )

        # -------------------------------------------------------------
        # 2. LEGAL RECOVERY: Imminent Default or Deep Overdue
        # -------------------------------------------------------------
        # Overdue >= 90 days
        if credit.days_overdue >= 90:
            return StrategyRecommendation(
                strategy=BankStrategy.LEGAL_RECOVERY,
                decision_code="LEGAL_DEEP_OVERDUE_90_PLUS",
                risk_level=RiskLevel.CRITICAL,
                confidence=0.99,
                is_automated=False,
                reasons=[
                    f"Глубокая просроченная задолженность: {credit.days_overdue} дней (порог NPL >= 90)",
                    f"Остаток неисполненных обязательств: {credit.current_disbursed:,.2f} руб.",
                    "Регулярные платежи прекращены",
                ],
                action_items=[
                    "Передать кредитное досье в Юридическое управление для судебного взыскания",
                    "Наложить арест на земельный участок и возведенные конструкции",
                    "Подготовить материалы для реализации обеспечения на публичных торгах",
                ],
                summary=(
                    f"ДЕФОЛТ: Просроченная задолженность {credit.days_overdue} дней. "
                    "Запуск процедуры принудительного судебного взыскания и реализации залога."
                ),
            )

        # Deadline < 90 days and 0% progress
        if days_to_deadline <= 90 and inspection.stage == StageCategory.STAGE_0:
            return StrategyRecommendation(
                strategy=BankStrategy.LEGAL_RECOVERY,
                decision_code="LEGAL_DEADLINE_EXPIRED_NO_PROGRESS",
                risk_level=RiskLevel.CRITICAL,
                confidence=0.95,
                is_automated=False,
                reasons=[
                    f"До нормативного срока ввода дома осталось {days_to_deadline} дней (< 90 дней)",
                    "Строительство фактически не начато (готовность 0% / участок пуст)",
                    "Физическая невозможность завершить возведение дома в установленный срок",
                ],
                action_items=[
                    "Инициировать досрочное истребование всей суммы кредита",
                    "Подготовить иск об обращении взыскания на земельный участок",
                    "Провести оценку рыночной стоимости участка для торгов",
                ],
                summary=(
                    f"ДЕФОЛТ СРОКА: До дедлайна {days_to_deadline} дн., готовность 0%. "
                    "Невозможность завершения стройки. Передача в суд на реализацию участка."
                ),
            )

        # -------------------------------------------------------------
        # 3. PLEDGE REGISTRATION: 100% Ready (Ready for Collateral)
        # -------------------------------------------------------------
        if inspection.stage == StageCategory.STAGE_100 or inspection.physical_readiness_pct >= 85.0:
            return StrategyRecommendation(
                strategy=BankStrategy.PLEDGE_REGISTRATION,
                decision_code="PLEDGE_REGISTRATION_READY",
                risk_level=RiskLevel.LOW,
                confidence=0.97,
                is_automated=False,
                reasons=[
                    f"Объект капитального строительства готов на {inspection.physical_readiness_pct:.1f}%",
                    "Тепловой контур здания полностью закрыт (фундамент, стены, кровля, окна/двери)",
                    "Объект соответствует критериям завершенного жилого дома для оформления залога",
                ],
                action_items=[
                    "Уведомить заемщика о необходимости подачи заявления на кадастровый учет и регистрацию залога",
                    "Оказать юридическое сопровождение через сервис электронной регистрации Сбера",
                    "Зафиксировать снижение ставки после регистрации закладной (если предусмотрено)",
                ],
                summary=(
                    f"РЕГИСТРАЦИЯ ЗАЛОГА: Строение готово ({inspection.physical_readiness_pct:.0f}%). "
                    "Необходимо оформить залог жилого дома в Росреестре."
                ),
            )

        # -------------------------------------------------------------
        # 4. RESTRUCTURING: Behind Schedule or Overdue
        # -------------------------------------------------------------
        has_active_overdue = 0 < credit.days_overdue < 90
        is_deadline_tight = days_to_deadline <= 180 and inspection.physical_readiness_pct < 60.0
        is_stalled = inspection.is_construction_abandoned

        if has_active_overdue or is_deadline_tight or is_stalled:
            reasons: List[str] = []
            if has_active_overdue:
                reasons.append(f"Зафиксирована просрочка платежей: {credit.days_overdue} дней")
            if is_deadline_tight:
                reasons.append(
                    f"Риск срыва срока ввода: осталось {days_to_deadline} дн. "
                    f"при готовности {inspection.physical_readiness_pct:.1f}%"
                )
            if is_stalled:
                reasons.append("Признаки приостановки строительных работ без консервации объекта")

            return StrategyRecommendation(
                strategy=BankStrategy.RESTRUCTURING,
                decision_code="RESTRUCTURING_PREVENTIVE",
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

        # -------------------------------------------------------------
        # 5. GREEN CORRIDOR: Tranche Approval (STP)
        # -------------------------------------------------------------
        return StrategyRecommendation(
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
                "Обновить статус объекта в АС «Калита» на 'Зеленый коридор'",
                "Запланировать следующий контрольный облет БПЛА через 60 дней",
            ],
            summary=(
                f"ЗЕЛЕНЫЙ КОРИДОР: Стройка в графике ({inspection.physical_readiness_pct:.0f}%), нарушений нет. "
                f"Автоматическое согласование транша №{credit.tranche_number} ({credit.next_tranche_amount:,.0f} руб.)."
            ),
        )
