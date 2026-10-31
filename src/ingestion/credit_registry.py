"""Bank Credit Registry repository for AgentDrone.

Provides access to borrower credit agreements, tranches, pledged values,
and declared construction parameters for the 13 hackathon objects.
"""

from __future__ import annotations

from ..models.schemas import CreditObject
from .cadastral_registry import normalize_cadastral_id

CREDIT_REGISTRY_DATA: dict[str, CreditObject] = {
    "54:07:055001:178": CreditObject(
        cadastral_id="54:07:055001:178",
        borrower_name="Смирнов Алексей Викторович",
        credit_agreement_number="КД-ИЖС-54/2026-0012",
        credit_amount_rub=5_800_000.0,
        planned_completion_date="2026-12-15",
        declared_wall_material="Кирпич",
        loan_purpose="ИЖС (Строительство жилого дома)",
        days_overdue=0,
        tranche_number=2,
        total_tranches=3,
        address="Новосибирская обл., Тогучинский р-н, с. Китерня, ул. Центральная",
    ),
    "54:19:130102:1231": CreditObject(
        cadastral_id="54:19:130102:1231",
        borrower_name="Кузнецов Михаил Андреевич",
        credit_agreement_number="КД-ИЖС-54/2026-0045",
        credit_amount_rub=7_200_000.0,
        planned_completion_date="2026-11-30",
        declared_wall_material="Газобетон/Блок",
        loan_purpose="ИЖС (Индивидуальный жилой дом)",
        days_overdue=0,
        tranche_number=2,
        total_tranches=3,
        address="Новосибирская обл., Новосибирский р-н, Кудряшовский сельсовет, п. Приобский",
    ),
    "54:19:130102:1495": CreditObject(
        cadastral_id="54:19:130102:1495",
        borrower_name="Васильева Ольга Николаевна",
        credit_agreement_number="КД-ИЖС-54/2026-0063",
        credit_amount_rub=6_500_000.0,
        planned_completion_date="2026-11-15",
        declared_wall_material="Газобетон/Блок",
        loan_purpose="ИЖС (Коттедж)",
        days_overdue=28,
        tranche_number=2,
        total_tranches=3,
        address="Новосибирская обл., Новосибирский р-н, Кудряшовский сельсовет",
    ),
    "54:19:160202:125": CreditObject(
        cadastral_id="54:19:160202:125",
        borrower_name="Федоров Дмитрий Сергеевич",
        credit_agreement_number="КД-ИЖС-54/2026-0081",
        credit_amount_rub=8_000_000.0,
        planned_completion_date="2026-10-01",
        declared_wall_material="Кирпич",
        loan_purpose="ИЖС (Усадьба)",
        days_overdue=0,
        tranche_number=3,
        total_tranches=3,
        address="Новосибирская обл., Новосибирский р-н, Барышевский сельсовет",
    ),
    "54:19:163001:521": CreditObject(
        cadastral_id="54:19:163001:521",
        borrower_name="Морозов Роман Игоревич",
        credit_agreement_number="КД-ЛПХ-54/2026-0099",
        credit_amount_rub=4_800_000.0,
        planned_completion_date="2026-12-01",
        declared_wall_material="Брус/Дерево",
        loan_purpose="ЛПХ (Жилой дом с надворными постройками)",
        days_overdue=14,
        tranche_number=2,
        total_tranches=3,
        address="Новосибирская обл., Новосибирский р-н, Морской сельсовет",
    ),
    "54:19:163001:661": CreditObject(
        cadastral_id="54:19:163001:661",
        borrower_name="Семенов Артем Владиславович",
        credit_agreement_number="КД-ИЖС-54/2026-0112",
        credit_amount_rub=9_500_000.0,
        planned_completion_date="2026-11-20",
        declared_wall_material="Кирпич",
        loan_purpose="ИЖС (Загородный дом)",
        days_overdue=0,
        tranche_number=2,
        total_tranches=3,
        address="Новосибирская обл., Новосибирский р-н, п. Ленинский",
    ),
    "54:19:164603:361": CreditObject(
        cadastral_id="54:19:164603:361",
        borrower_name="Павлов Сергей Павлович",
        credit_agreement_number="КД-ИЖС-54/2026-0134",
        credit_amount_rub=6_200_000.0,
        planned_completion_date="2026-12-25",
        declared_wall_material="Газобетон/Блок",
        loan_purpose="ИЖС (Жилой дом)",
        days_overdue=0,
        tranche_number=1,
        total_tranches=3,
        address="Новосибирская обл., Новосибирский р-н, с. Боровое",
    ),
    "54:19:164603:3847": CreditObject(
        cadastral_id="54:19:164603:3847",
        borrower_name="Ковалев Андрей Юрьевич",
        credit_agreement_number="КД-ИЖС-54/2026-0158",
        credit_amount_rub=7_600_000.0,
        planned_completion_date="2026-10-15",
        declared_wall_material="Кирпич",
        loan_purpose="ИЖС (Жилой дом)",
        days_overdue=45,
        tranche_number=2,
        total_tranches=3,
        address="Новосибирская обл., Новосибирский р-н, п. Береговое",
    ),
    "54:19:165001:694": CreditObject(
        cadastral_id="54:19:165001:694",
        borrower_name="Егорова Татьяна Анатольевна",
        credit_agreement_number="КД-ИЖС-54/2026-0177",
        credit_amount_rub=6_900_000.0,
        planned_completion_date="2026-12-10",
        declared_wall_material="Газобетон/Блок",
        loan_purpose="ИЖС (Жилой дом)",
        days_overdue=0,
        tranche_number=2,
        total_tranches=3,
        address="Новосибирская обл., Новосибирский р-н, с. Новопичугово",
    ),
    "54:19:165001:729": CreditObject(
        cadastral_id="54:19:165001:729",
        borrower_name="Белов Илья Константинович",
        credit_agreement_number="КД-ИЖС-54/2026-0192",
        credit_amount_rub=6_100_000.0,
        planned_completion_date="2026-11-05",
        declared_wall_material="Брус/Дерево",
        loan_purpose="ИЖС (Дом у воды)",
        days_overdue=0,
        tranche_number=3,
        total_tranches=3,
        address="Новосибирская обл., Новосибирский р-н, с. Новопичугово, ул. Набережная",
    ),
    "54:32:010081:150": CreditObject(
        cadastral_id="54:32:010081:150",
        borrower_name="Григорьев Виктор Николаевич",
        credit_agreement_number="КД-ИЖС-54/2026-0210",
        credit_amount_rub=10_500_000.0,
        planned_completion_date="2026-08-01",
        declared_wall_material="Кирпич",
        loan_purpose="ИЖС (Коттедж)",
        days_overdue=115,  # Long overdue / Foreclosure scenario candidate
        tranche_number=2,
        total_tranches=3,
        address="Новосибирская обл., г. Бердск, мкр. Южный",
    ),
    "54:32:010870:8": CreditObject(
        cadastral_id="54:32:010870:8",
        borrower_name="ООО «СибТоргСервис» (Бердск)",
        credit_agreement_number="КД-ИЖС-54/2026-0235",
        credit_amount_rub=12_000_000.0,
        planned_completion_date="2026-10-30",
        declared_wall_material="Кирпич (жилой дом)",
        loan_purpose="ИЖС (по кредитному договору заявлен индивидуальный жилой дом)",
        days_overdue=0,
        tranche_number=2,
        total_tranches=3,
        address="Новосибирская обл., г. Бердск, ул. Энгельса, д. 4",
    ),
    "54:35:121015:124": CreditObject(
        cadastral_id="54:35:121015:124",
        borrower_name="Соколова Екатерина Михайловна",
        credit_agreement_number="КД-ИЖС-54/2026-0255",
        credit_amount_rub=8_900_000.0,
        planned_completion_date="2026-11-28",
        declared_wall_material="Кирпич",
        loan_purpose="ИЖС (Коттедж Академгородок)",
        days_overdue=0,
        tranche_number=2,
        total_tranches=3,
        address="г. Новосибирск, Советский р-н (Академгородок)",
    ),
}


class CreditRegistry:
    """Service to access credit agreements for monitored collateral plots."""

    @classmethod
    def get_credit_object(cls, identifier: str) -> CreditObject:
        """Find credit object by cadastral ID or raw numeric folder name."""
        norm_id = normalize_cadastral_id(identifier)
        if norm_id in CREDIT_REGISTRY_DATA:
            return CREDIT_REGISTRY_DATA[norm_id]

        # Fallback default object
        return CreditObject(
            cadastral_id=norm_id,
            borrower_name="Заемщик ПАО Сбербанк",
            credit_agreement_number=f"КД-54/{norm_id.replace(':', '')[-4:]}",
            credit_amount_rub=6_000_000.0,
            planned_completion_date="2026-12-31",
            declared_wall_material="Кирпич",
            loan_purpose="ИЖС",
            days_overdue=0,
            tranche_number=2,
            total_tranches=3,
            address="Новосибирская область",
        )

    @classmethod
    def list_all_credits(cls) -> list[CreditObject]:
        """Return all credit agreements in the portfolio."""
        return list(CREDIT_REGISTRY_DATA.values())
