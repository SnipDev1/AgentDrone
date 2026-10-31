"""GigaChat API integration service for Siberian Bank AS 'Kalita' reporting.

Provides an AI-powered expert inspector summary using Sber GigaChat SDK
with a deterministic, professional offline fallback generator.
Supports both banking domain models and CV inspection pipeline schemas.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .prompts import SYSTEM_INSPECTOR_PROMPT, build_inspector_prompt, extract_context_payload

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
            self._client = None
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

    def set_credentials(
        self,
        credentials: str,
        scope: Optional[str] = None,
        model: Optional[str] = None,
    ) -> bool:
        """Re-initialize client with runtime credentials (e.g. from Streamlit UI)."""
        self.credentials = credentials.strip()
        if scope:
            self.scope = scope
        if model:
            self.model = model
        self._init_client()
        return self.is_online

    @property
    def is_online(self) -> bool:
        """Check if active GigaChat client is available."""
        return self._is_online and self._client is not None

    def get_mode_status(self) -> dict[str, Any]:
        """Return human-readable service status for UI indicators."""
        return {
            "is_online": self.is_online,
            "mode": "Онлайн (GigaChat API Сбер)" if self.is_online else "Оффлайн (Шаблонизатор Сбера)",
            "model": self.model if self.is_online else "Offline-Fallback-Engine",
            "scope": self.scope,
            "has_credentials": bool(self.credentials),
        }

    def generate_summary(
        self,
        credit: Any,
        inspection: Any,
        recommendation: Any,
    ) -> str:
        """Generate expert inspector summary (via API or offline fallback)."""
        payload = extract_context_payload(credit, inspection, recommendation)

        if self.is_online and self._client:
            try:
                user_content = build_inspector_prompt(credit, inspection, recommendation)
                response = self._client.chat({
                    "messages": [
                        {"role": "system", "content": SYSTEM_INSPECTOR_PROMPT},
                        {"role": "user", "content": user_content},
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
        return self._generate_offline_summary(payload)

    def generate_summary_for_ui(
        self,
        cv_result: Any,
        credit_obj: Any,
        decision: Any,
    ) -> str:
        """Convenience method for Streamlit UI passing CV pipeline schemas."""
        return self.generate_summary(credit=credit_obj, inspection=cv_result, recommendation=decision)

    def _generate_offline_summary(self, payload: dict[str, Any]) -> str:
        """Generate a deterministic, legally accurate bank summary for AS 'Kalita'."""
        credit_info = payload.get("кредитный_договор", {})
        insp_info = payload.get("результаты_бпла_осмотра", {})
        dec_info = payload.get("решение_decision_engine", {})

        cadastral = credit_info.get("кадастровый_номер", "")
        borrower = credit_info.get("заемщик", "Заемщик")
        overdue = credit_info.get("просрочка_дней", 0)
        proj_mat = credit_info.get("проектный_материал", "")
        deadline = credit_info.get("дедлайн_стройки", "")

        date_str = insp_info.get("дата_съемки", "")
        readiness_pct = float(insp_info.get("физическая_готовность_процент", 0.0))
        det_mat = insp_info.get("фактический_материал", "")
        is_commercial = insp_info.get("риск_коммерческого_использования", False)
        mat_match = insp_info.get("соответствие_материала", True)

        strat_str = str(dec_info.get("стратегия", ""))
        strat_code = str(dec_info.get("код_решения", ""))

        header = f"ЗАКЛЮЧЕНИЕ ИНСПЕКТОРА ПО ОБЪЕКТУ {cadastral}:\n"

        # 1. STOP-FRAUD
        if "FRAUD" in strat_code or "Блокировка" in strat_str or is_commercial or not mat_match:
            if is_commercial:
                return (
                    f"{header}"
                    f"В ходе аэромониторинга БПЛА от {date_str} на участке ИЖС "
                    "зафиксировано возведение нежилого коммерческого объекта (склад / супермаркет / автосервис). "
                    "Выявлено прямое нарушение целевого назначения земельного участка. "
                    "В соответствии со ст. 813, 814 ГК РФ выдача траншей заблокирована, кредитное досье передано "
                    "в Управление безопасности Сибирского банка для проведения служебного расследования."
                )
            else:
                return (
                    f"{header}"
                    f"По результатам обследования БПЛА выявлено критическое расхождение фактического материала несущих стен "
                    f"({det_mat}) с утвержденной проектно-сметной документацией ({proj_mat}). "
                    "Имеются признаки умышленного удешевления сметной стоимости предмета залога. "
                    "Кредитование временно приостановлено. Назначена строительно-техническая экспертиза сметы."
                )

        # 2. DEFAULT & FORECLOSURE
        if "DEFAULT" in strat_code or "LEGAL" in strat_code or "Дефолт" in strat_str or "Взыскание" in strat_str:
            return (
                f"{header}"
                f"Объект находится в зоне дефолтного риска. При нормативном сроке ввода до {deadline} "
                f"фактическая строительная готовность строения составляет {readiness_pct:.1f}% (просрочка {overdue} дн.). "
                "Физическая возможность завершения строительства в договорные сроки отсутствует. "
                "Рекомендован запуск процедуры досрочного истребования задолженности и судебного обращения взыскания "
                "на обеспечение для последующей реализации на торгах."
            )

        # 3. PLEDGE REGISTRATION
        if "REGISTER" in strat_code or "PLEDGE" in strat_code or "Регистрация" in strat_str:
            return (
                f"{header}"
                f"Строительство жилого дома успешно завершено (фактическая готовность {readiness_pct:.1f}%, "
                "тепловой контур полностью закрыт). Объект готов к оформлению в качестве постоянного обеспечения Банка. "
                f"Заемщику {borrower} направлено официальное уведомление о необходимости подачи документов "
                "на регистрацию права собственности и залога жилого дома в пользу Сибирского банка ПАО Сбербанк."
            )

        # 4. RESTRUCTURING
        if "RESTRUCTURE" in strat_code or "Реструктуризация" in strat_str or overdue > 0:
            return (
                f"{header}"
                f"Зафиксировано отставание от проектного графика строительства (фактическая готовность {readiness_pct:.1f}%, "
                f"просрочка платежей {overdue} дн., срок завершения: {deadline}). "
                "В целях недопущения перехода кредита в категорию NPL и защиты финансового положения заемщика "
                "рекомендована программа реструктуризации: предоставление кредитных каникул и пролонгация нормативного срока "
                "строительства с утверждением скорректированного графика траншей."
            )

        # 5. TRANCHE APPROVAL
        return (
            f"{header}"
            "Строительство объекта ведется в строгом соответствии с графиком кредитного соглашения и проектной сметой. "
            f"Фактическая готовность {readiness_pct:.1f}% подтверждает выполнение контрольного этапа. Несущие стены "
            f"выполнены из проектного материала ({proj_mat}), градостроительные отступы соблюдены, просрочка отсутствует. "
            "Автоматически согласован очередной транш в режиме сквозной обработки STP без созыва кредитного комитета."
        )
