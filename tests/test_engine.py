"""Unit tests for Siberian Bank 5-Strategy Decision Engine (TB-04, ENG-01).

Verifies 100% rule coverage for all 5 banking strategies and boundary conditions:
1. Tranche Approval (Green Corridor, STP)
2. Restructuring (Financial Recovery)
3. Collateral Pledge Registration
4. Stop-Fraud / Security Audit (Commercial use, Material substitution)
5. Legal Enforcement / Foreclosure (Default, Deep overdue)

Also verifies compatibility with CVInspectionResult / Streamlit UI contracts.
"""

from datetime import date
import pytest

from src.engine.decision_engine import DecisionEngine
from src.models import (
    CreditObject,
    InspectionResult,
    BuildingElements,
    BankStrategy,
    RiskLevel,
    StageCategory,
    WallMaterial,
)
from src.models.schemas import (
    CadastralGeoCheck,
    CreditObject as CVCreditObject,
    CVInspectionResult,
    StrategyType,
    WallMaterial as CVWallMaterial,
)


# =============================================================================
# Domain Model Tests (Andrey's Suite - 9 tests)
# =============================================================================

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
    base_credit.construction_deadline = date(2026, 11, 8)
    base_inspection.physical_readiness_pct = 0.0
    base_inspection.stage = StageCategory.STAGE_0

    engine = DecisionEngine()
    decision = engine.evaluate(base_credit, base_inspection, eval_date=date(2026, 9, 24))

    assert decision.strategy == BankStrategy.LEGAL_RECOVERY
    assert decision.risk_level == RiskLevel.CRITICAL
    assert decision.decision_code == "LEGAL_IMMINENT_DEADLINE_DEFAULT"


def test_strategy_3_pledge_registration_at_100_percent(base_credit, base_inspection):
    """Scenario 3: Construction 100% finished (thermal envelope closed) -> Mortgage registration."""
    base_inspection.elements = BuildingElements(
        foundation=True, walls=True, roof=True, windows_doors=True
    )
    base_inspection.physical_readiness_pct = 100.0
    base_inspection.stage = StageCategory.STAGE_100

    engine = DecisionEngine()
    decision = engine.evaluate(base_credit, base_inspection)

    assert decision.strategy == BankStrategy.PLEDGE_REGISTRATION
    assert decision.risk_level == RiskLevel.LOW
    assert decision.is_automated is True
    assert decision.decision_code == "PLEDGE_REGISTRATION_READY"
    assert any("ипотек" in action.lower() for action in decision.action_items)


def test_strategy_2_restructuring_overdue(base_credit, base_inspection):
    """Scenario 2a: Overdue 30 days (<90) -> Financial recovery / Restructuring."""
    base_credit.days_overdue = 30

    engine = DecisionEngine()
    decision = engine.evaluate(base_credit, base_inspection)

    assert decision.strategy == BankStrategy.RESTRUCTURING
    assert decision.decision_code == "RESTRUCTURING_OVERDUE_WARNING"
    assert any("каникул" in action.lower() for action in decision.action_items)


def test_strategy_2_restructuring_tight_deadline_behind_schedule(base_credit, base_inspection):
    """Scenario 2b: Deadline in 120 days, but still Stage 0 -> Restructure timeline."""
    base_credit.construction_deadline = date(2027, 1, 20)
    base_inspection.stage = StageCategory.STAGE_0
    base_inspection.physical_readiness_pct = 10.0

    engine = DecisionEngine()
    decision = engine.evaluate(base_credit, base_inspection, eval_date=date(2026, 9, 24))

    assert decision.strategy == BankStrategy.RESTRUCTURING
    assert decision.decision_code == "RESTRUCTURING_BEHIND_SCHEDULE"


def test_priority_hierarchy_fraud_overrides_100_percent_ready(base_credit, base_inspection):
    """Verify STOP-FRAUD has absolute priority even if physical building is 100% complete."""
    base_inspection.stage = StageCategory.STAGE_100
    base_inspection.physical_readiness_pct = 100.0
    base_inspection.is_commercial_building_risk = True

    engine = DecisionEngine()
    decision = engine.evaluate(base_credit, base_inspection)

    assert decision.strategy == BankStrategy.FRAUD_ALERT
    assert decision.risk_level == RiskLevel.CRITICAL


# =============================================================================
# CVInspectionResult & UI Tests (Dmitry's Suite - 5 tests)
# =============================================================================

def _create_mock_cv_result(
    readiness_pct: float = 50.0,
    readiness_stage: int = 50,
    fraud_risk: bool = False,
    fraud_reasons: list[str] | None = None,
    material: CVWallMaterial = CVWallMaterial.BRICK,
    is_correct_plot: bool = True,
    distance_to_cadastre: float = 12.0,
) -> CVInspectionResult:
    return CVInspectionResult(
        cadastral_id="54:19:130102:1231",
        readiness_pct=readiness_pct,
        readiness_stage=readiness_stage,
        stage_title=f"Стадия {readiness_stage}%",
        detected_material=material,
        fraud_risk_flag=fraud_risk,
        fraud_reasons=fraud_reasons or [],
        geocheck=CadastralGeoCheck(
            cadastral_id="54:19:130102:1231",
            drone_center_lat=55.057,
            drone_center_lon=83.147,
            distance_to_cadastre_m=distance_to_cadastre,
            is_correct_plot=is_correct_plot,
        ),
    )


def _create_mock_credit_obj(
    days_overdue: int = 0,
    tranche_number: int = 2,
    credit_amount: float = 6_000_000.0,
) -> CVCreditObject:
    return CVCreditObject(
        cadastral_id="54:19:130102:1231",
        borrower_name="Иванов Иван Иванович",
        credit_agreement_number="КД-2026-54-102",
        credit_amount_rub=credit_amount,
        planned_completion_date="2026-12-01",
        declared_wall_material="Кирпич",
        days_overdue=days_overdue,
        tranche_number=tranche_number,
    )


def test_strat_01_approve_tranche():
    """Test standard normal tranche approval with CV result."""
    cv = _create_mock_cv_result(readiness_pct=55.0, readiness_stage=50)
    credit = _create_mock_credit_obj(days_overdue=0, tranche_number=2)

    rec = DecisionEngine.evaluate(cv, credit)
    assert rec.strategy == StrategyType.APPROVE_TRANCHE
    assert rec.strategy_code == "STRAT-01"
    assert rec.risk_level == "LOW"
    assert rec.kalita_status == "APPROVED_AUTO"


def test_strat_02_restructure_extend():
    """Test moderate delay triggering restructuring / extension."""
    cv = _create_mock_cv_result(readiness_pct=42.0, readiness_stage=50)
    credit = _create_mock_credit_obj(days_overdue=35, tranche_number=2)

    rec = DecisionEngine.evaluate(cv, credit)
    assert rec.strategy == StrategyType.RESTRUCTURE_EXTEND
    assert rec.strategy_code == "STRAT-02"
    assert rec.risk_level == "MEDIUM"
    assert rec.kalita_status == "MANUAL_REVIEW"


def test_strat_03_register_collateral():
    """Test 100% completed building ready for mortgage registration."""
    cv = _create_mock_cv_result(readiness_pct=96.0, readiness_stage=100)
    credit = _create_mock_credit_obj(days_overdue=0, tranche_number=3)

    rec = DecisionEngine.evaluate(cv, credit)
    assert rec.strategy == StrategyType.REGISTER_COLLATERAL
    assert rec.strategy_code == "STRAT-03"
    assert rec.risk_level == "LOW"
    assert rec.kalita_status == "COLLATERAL_REGISTERED"


def test_strat_04_fraud_suspension():
    """Test fraud trigger (e.g. commercial building or material substitute)."""
    cv = _create_mock_cv_result(
        readiness_pct=75.0,
        fraud_risk=True,
        fraud_reasons=["Нецелевое использование: супермаркет на участке ИЖС"],
    )
    credit = _create_mock_credit_obj(days_overdue=0, tranche_number=2)

    rec = DecisionEngine.evaluate(cv, credit)
    assert rec.strategy == StrategyType.SUSPEND_SECURITY_FRAUD
    assert rec.strategy_code == "STRAT-04"
    assert rec.risk_level == "CRITICAL"
    assert rec.kalita_status == "BLOCKED_FRAUD"


def test_strat_05_default_foreclosure():
    """Test abandoned construction with >90 days overdue triggering foreclosure."""
    cv = _create_mock_cv_result(readiness_pct=10.0, readiness_stage=0)
    credit = _create_mock_credit_obj(days_overdue=120, tranche_number=2)

    rec = DecisionEngine.evaluate(cv, credit)
    assert rec.strategy == StrategyType.DEFAULT_FORECLOSURE
    assert rec.strategy_code == "STRAT-05"
    assert rec.risk_level == "CRITICAL"
    assert rec.kalita_status == "LEGAL_ACTION"
