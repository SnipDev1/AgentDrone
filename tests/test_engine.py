"""Unit tests for the Bank Decision Engine."""

from src.engine.decision_engine import DecisionEngine
from src.models.schemas import (
    CadastralGeoCheck,
    CreditObject,
    CVInspectionResult,
    StrategyType,
    WallMaterial,
)


def _create_mock_cv_result(
    readiness_pct: float = 50.0,
    readiness_stage: int = 50,
    fraud_risk: bool = False,
    fraud_reasons: list[str] | None = None,
    material: WallMaterial = WallMaterial.BRICK,
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
) -> CreditObject:
    return CreditObject(
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
    """Test standard normal tranche approval."""
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
