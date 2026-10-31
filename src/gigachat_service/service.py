"""GigaChat API integration service for Siberian Bank AS 'Kalita' reporting.

Provides an AI-powered expert inspector summary using Sber GigaChat SDK
with a deterministic, professional offline fallback generator.
"""

import os
import logging
from typing import Optional

from ..models import CreditObject, InspectionResult, StrategyRecommendation, BankStrategy
from .prompts import SYSTEM_INSPECTOR_PROMPT, build_inspector_prompt

logger = logging.getLogger(__name__)


class GigaChatInspectorService:
    """Generates official inspection summaries via GigaChat API with offline fallback."""

    def __init__(
        self,
        credentials: Optional[str] = None,
        scope: Optional[str] = None,
        model: str = "GigaChat:latest",
        verify_ssl_certs: bool = False,
    ):
        self.credentials = credentials or os.getenv("GIGACHAT_CREDENTIALS")
        self.scope = scope or os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self.model = model or os.getenv("GIGACHAT_MODEL", "GigaChat:latest")
        self.verify_ssl_certs = verify_ssl_certs
        self._client = None
        self._is_online = False
        self._init_client()

    def _init_client(self) -> None:
        """Attempt to initialize GigaChat client if credentials exist."""
        if not self.credentials:
            logger.info("GigaChat credentials not provided. Operating in offline template fallback mode.")
            self._is_online = False
            return

        try:
            from gigachat import GigaChat
            self._client = GigaChat(
                credentials=self.credentials,
                scope=self.scope,
                model=self.model,
                verify_ssl_certs=self.verify_ssl_certs,
            )
            self._is_online = True
            logger.info("GigaChat client successfully initialized.")
        except Exception as e:
            logger.warning(f"Failed to initialize GigaChat client: {e}. Falling back to offline mode.")
            self._client = None
            self._is_online = False

    @property
    def is_online(self) -> bool:
        """Check if active GigaChat client is available."""
        return self._is_online and self._client is not None

    def generate_summary(
        self,
        credit: CreditObject,
        inspection: InspectionResult,
        recommendation: StrategyRecommendation
    ) -> str:
        """Generate expert inspector summary (via API or offline fallback)."""
        if self.is_online and self._client:
            try:
                user_content = build_inspector_prompt(credit, inspection, recommendation)
                response = self._client.chat({
                    "messages": [
                        {"role": "system", "content": SYSTEM_INSPECTOR_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 600,
                })
                if response and response.choices and len(response.choices) > 0:
                    text = response.choices[0].message.content.strip()
                    if text:
                        return text
            except Exception as e:
                logger.warning(f"GigaChat API call failed: {e}. Generating offline expert summary.")

        # Offline fallback
        return self._generate_offline_summary(credit, inspection, recommendation)

    def _generate_offline_summary(
        self,
        credit: CreditObject,
        inspection: InspectionResult,
        recommendation: StrategyRecommendation
    ) -> str:
        """Generate a deterministic, legally accurate bank summary for AS 'Kalita'."""
        days_to_deadline = (credit.construction_deadline - inspection.inspection_date).days
        strat = recommendation.strategy

        if strat == BankStrategy.FRAUD_ALERT:
            if inspection.is_commercial_building_risk:
                return (
                    f"ЗАКЛЮЧЕНИЕ ИНСПЕКТОРА ПО ОБЪЕКТУ {credit.cadastral_number}:\n"
                    f"В ходе аэромониторинга БПЛА от {inspection.inspection_date.strftime('%d.%m.%Y')} на участке ИЖС "
                    f"зафиксировано возведение нежилого коммерческого здания. Вид разрешенного использования земли "
                    f"'{credit.land_category_vri}' нарушен. Назначена проверка Службы безопасности Банка. "
                    f"Выдача траншей заблокирована до устранения нецелевого использования."
                )
            else:
                return (
                    f"ЗАКЛЮЧЕНИЕ ИНСПЕКТОРА ПО ОБЪЕКТУ {credit.cadastral_number}:\n"
                    f"По результатам обследования выявлено расхождение фактического материала стен "
                    f"({inspection.detected_wall_material.value}) с проектно-сметной документацией "
                    f"({credit.projected_wall_material.value}). Риск удешевления залоговой стоимости. "
                    f"Кредитование приостановлено для проведения аудита сметы."
                )

        elif strat == BankStrategy.LEGAL_RECOVERY:
            return (
                f"ЗАКЛЮЧЕНИЕ ИНСПЕКТОРА ПО ОБЪЕКТУ {credit.cadastral_number}:\n"
                f"Объект находится в зоне критического риска (дефолт). При сроке договора до "
                f"{credit.construction_deadline.strftime('%d.%m.%Y')} (осталось {days_to_deadline} дн.) "
                f"строительная готовность составляет {inspection.physical_readiness_pct:.0f}% "
                f"(просрочка: {credit.days_overdue} дн.). Физическая возможность завершения стройки отсутствует. "
                f"Материалы переданы в Юридическое управление для судебного обращения взыскания на земельный участок."
            )

        elif strat == BankStrategy.PLEDGE_REGISTRATION:
            return (
                f"ЗАКЛЮЧЕНИЕ ИНСПЕКТОРА ПО ОБЪЕКТУ {credit.cadastral_number}:\n"
                f"Строительство жилого дома полностью завершено ({inspection.physical_readiness_pct:.0f}%, "
                f"тепловой контур закрыт). Объект готов к оформлению в качестве постоянного обеспечения Банка. "
                f"Заемщику {credit.borrower_name} направлено предписание об оформлении права собственности "
                f"и регистрации залога в пользу Сибирского банка ПАО Сбербанк через сервис электронной регистрации."
            )

        elif strat == BankStrategy.RESTRUCTURING:
            return (
                f"ЗАКЛЮЧЕНИЕ ИНСПЕКТОРА ПО ОБЪЕКТУ {credit.cadastral_number}:\n"
                f"Зафиксировано отставание от графика (готовность {inspection.physical_readiness_pct:.0f}%, "
                f"просрочка {credit.days_overdue} дн., до дедлайна {days_to_deadline} дн.). "
                "В целях недопущения дефолта заемщику рекомендована программа финансового оздоровления: "
                "предоставление кредитных каникул и пролонгация срока строительства для завершения коробки здания."
            )

        else:  # BankStrategy.TRANCHE_APPROVAL
            return (
                f"ЗАКЛЮЧЕНИЕ ИНСПЕКТОРА ПО ОБЪЕКТУ {credit.cadastral_number}:\n"
                "Строительство объекта ведется в строгом соответствии с проектной сметой и графиком финансирования. "
                f"Фактическая готовность составляет {inspection.physical_readiness_pct:.0f}%, несущие конструкции "
                f"выполнены из проектного материала ({credit.projected_wall_material.value}), "
                "градостроительные отступы соблюдены. Просроченная задолженность отсутствует. "
                f"Автоматически согласован очередной транш №{credit.tranche_number} "
                f"в размере {credit.next_tranche_amount:,.0f} руб."
            )
