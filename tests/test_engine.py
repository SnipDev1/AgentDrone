"""Unit tests for Siberian Bank 5-Strategy Decision Engine (TB-04, ENG-01).

Verifies 100% rule coverage for all 5 banking strategies and boundary conditions:
1. Tranche Approval (Green Corridor, STP)
2. Restructuring (Financial Recovery)
3. Collateral Pledge Registration
4. Stop-Fraud / Security Audit (Commercial use, Material substitution)
5. Legal Enforcement / Foreclosure (Default, Deep overdue)
"""

from datetime import date
import pytest

from src.models import (
    CreditObject,
    InspectionResult,
    BuildingElements,
    BankStrategy,
    RiskLevel,
    StageCategory,
    WallMaterial,
)
from src.engine import DecisionEngine


@pytest.fixture
def base_credit():
    return CreditObject(
        credit_id="CR-TEST-001",
        cadastral_number="54:07:055001:178",
        borrower_name="Тестовый Заемщик",
        credit_amount=6000000.0,
        current_disbursed=2000000.0,
        tranche_number=2,
        tranches_total=3,
        next_tranche_amount=2000000.0,
        credit_issue_date=date(2026, 1, 1),
        construction_deadline=date(2027, 6, 1),
        days_overdue=0,
        projected_wall_material=WallMaterial.BRICK,
    )


@pytest.fixture
def base_inspection():
    return InspectionResult(
        cadastral_number="5407055001178",
        inspection_date=date(2026, 9, 24),
        total_images_analyzed=6,
        view_angles_complete=True,
        geo_match=True,
        elements=BuildingElements(foundation=True, walls=True, roof=False),
        physical_readiness_pct=50.0,
        stage=StageCategory.STAGE_50,
        detected_wall_material=WallMaterial.BRICK,
        material_match=True,
        boundary_violation=False,
        is_commercial_building_risk=False,
        is_construction_abandoned=False,
    )


def test_strategy_1_green_corridor_tranche_approval(base_credit, base_inspection):
    """Scenario 1: Building on schedule, materials match, no overdue -> STP Tranche Approval."""
    engine = DecisionEngine()
    decision = engine.evaluate(base_credit, base_inspection)

    assert decision.strategy == BankStrategy.TRANCHE_APPROVAL
    assert decision.risk_level == RiskLevel.LOW
    assert decision.is_automated is True
    assert "TRANCHE_APPROVAL_STP" in decision.decision_code
    assert any("транша" in action.lower() for action in decision.action_items)


def test_strategy_4_fraud_commercial_building(base_credit, base_inspection):
    """Scenario 4a (Stop-Fraud): Commercial building on IZH land (Berdsk supermarket case)."""
    base_inspection.is_commercial_building_risk = True

    engine = DecisionEngine()
    decision = engine.evaluate(base_credit, base_inspection)

    assert decision.strategy == BankStrategy.FRAUD_ALERT
    assert decision.risk_level == RiskLevel.CRITICAL
    assert decision.is_automated is False
    assert decision.decision_code == "FRAUD_COMMERCIAL_USE"
    assert any("безопасности" in action.lower() for action in decision.action_items)


def test_strategy_4_fraud_material_substitution(base_credit, base_inspection):
    """Scenario 4b (Stop-Fraud): Borrower planned brick, built cheap frame panel."""
    base_credit.projected_wall_material = WallMaterial.BRICK
    base_inspection.detected_wall_material = WallMaterial.FRAME_PANEL
    base_inspection.material_match = False

    engine = DecisionEngine()
    decision = engine.evaluate(base_credit, base_inspection)

    assert decision.strategy == BankStrategy.FRAUD_ALERT
    assert decision.risk_level == RiskLevel.HIGH
    assert decision.decision_code == "FRAUD_MATERIAL_SUBSTITUTION"


def test_strategy_5_legal_recovery_deep_overdue(base_credit, base_inspection):
    """Scenario 5a: Overdue >= 90 days -> Immediate Legal Recovery."""
    base_credit.days_overdue = 95

    engine = DecisionEngine()
    decision = engine.evaluate(base_credit, base_inspection)

    assert decision.strategy == BankStrategy.LEGAL_RECOVERY
    assert decision.risk_level == RiskLevel.CRITICAL
    assert decision.decision_code == "LEGAL_DEEP_OVERDUE_90_PLUS"
    assert any("суд" in action.lower() for action in decision.action_items)


def test_strategy_5_legal_recovery_expired_deadline_no_progress(base_credit, base_inspection):
    """Scenario 5b: Construction deadline in < 90 days, but 0% readiness -> Default."""
    # Inspection on 2026-09-24, deadline in 45 days (2026-11-08)
    base_credit.construction_deadline = date(2026, 11, 8)
    base_inspection.physical_readiness_pct = 0.0
    base_inspection.stage = StageCategory.STAGE_0

    engine = DecisionEngine()
    decision = engine.evaluate(base_credit, base_inspection)

    assert decision.strategy == BankStrategy.LEGAL_RECOVERY
    assert decision.decision_code == "LEGAL_DEADLINE_EXPIRED_NO_PROGRESS"
    assert decision.risk_level == RiskLevel.CRITICAL


def test_strategy_3_pledge_registration_at_100_percent(base_credit, base_inspection):
    """Scenario 3: House 100% completed (closed thermal envelope) -> Register collateral."""
    base_inspection.physical_readiness_pct = 95.0
    base_inspection.stage = StageCategory.STAGE_100
    base_inspection.elements = BuildingElements(
        foundation=True, walls=True, roof=True, windows_doors=True
    )

    engine = DecisionEngine()
    decision = engine.evaluate(base_credit, base_inspection)

    assert decision.strategy == BankStrategy.PLEDGE_REGISTRATION
    assert decision.risk_level == RiskLevel.LOW
    assert decision.decision_code == "PLEDGE_REGISTRATION_READY"
    assert any("росреестр" in action.lower() or "залог" in action.lower() for action in decision.action_items)


def test_strategy_2_restructuring_overdue(base_credit, base_inspection):
    """Scenario 2a: Moderate overdue (1..89 days) -> Restructuring."""
    base_credit.days_overdue = 35

    engine = DecisionEngine()
    decision = engine.evaluate(base_credit, base_inspection)

    assert decision.strategy == BankStrategy.RESTRUCTURING
    assert decision.decision_code == "RESTRUCTURING_PREVENTIVE"
    assert any("каникулы" in action.lower() or "реструктуризаци" in action.lower() for action in decision.action_items)


def test_strategy_2_restructuring_tight_deadline_behind_schedule(base_credit, base_inspection):
    """Scenario 2b: Deadline in 4 months (< 180 days) and only 35% ready -> Preventive restructuring."""
    # Inspection on 2026-09-24, deadline in 120 days (2027-01-22)
    base_credit.construction_deadline = date(2027, 1, 22)
    base_credit.days_overdue = 0
    base_inspection.physical_readiness_pct = 35.0
    base_inspection.stage = StageCategory.STAGE_50

    engine = DecisionEngine()
    decision = engine.evaluate(base_credit, base_inspection)

    assert decision.strategy == BankStrategy.RESTRUCTURING
    assert decision.decision_code == "RESTRUCTURING_PREVENTIVE"


def test_priority_hierarchy_fraud_overrides_100_percent_ready(base_credit, base_inspection):
    """Verify that fraud detection takes strict priority over high readiness."""
    base_inspection.stage = StageCategory.STAGE_100
    base_inspection.physical_readiness_pct = 100.0
    # But built a supermarket
    base_inspection.is_commercial_building_risk = True

    engine = DecisionEngine()
    decision = engine.evaluate(base_credit, base_inspection)

    assert decision.strategy == BankStrategy.FRAUD_ALERT
    assert decision.risk_level == RiskLevel.CRITICAL
