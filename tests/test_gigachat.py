"""Unit tests for GigaChat API integration and offline fallback (ENG-02).

Verifies prompt formatting, offline fallback templates across all 5 strategies,
and mock-tested GigaChat API invocation.
"""

from datetime import date
from unittest.mock import MagicMock
import pytest

from src.models import (
    CreditObject,
    InspectionResult,
    StrategyRecommendation,
    BankStrategy,
    RiskLevel,
    StageCategory,
    WallMaterial,
)
from src.gigachat_service import (
    GigaChatInspectorService,
    build_inspector_prompt,
)


@pytest.fixture
def sample_credit():
    return CreditObject(
        credit_id="CR-GIGA-001",
        cadastral_number="54:07:055001:178",
        borrower_name="Сидоров Андрей Владимирович",
        credit_amount=6200000.0,
        current_disbursed=2500000.0,
        tranche_number=2,
        tranches_total=3,
        next_tranche_amount=2000000.0,
        credit_issue_date=date(2026, 2, 10),
        construction_deadline=date(2027, 4, 30),
        days_overdue=0,
        projected_wall_material=WallMaterial.BRICK,
    )


@pytest.fixture
def sample_inspection():
    return InspectionResult(
        cadastral_number="5407055001178",
        inspection_date=date(2026, 9, 24),
        total_images_analyzed=6,
        view_angles_complete=True,
        geo_match=True,
        physical_readiness_pct=50.0,
        stage=StageCategory.STAGE_50,
        detected_wall_material=WallMaterial.BRICK,
        material_match=True,
    )


def test_build_inspector_prompt(sample_credit, sample_inspection):
    """Verify structured prompt building contains all relevant banking info."""
    rec = StrategyRecommendation(
        strategy=BankStrategy.TRANCHE_APPROVAL,
        decision_code="TRANCHE_APPROVAL_STP",
        risk_level=RiskLevel.LOW,
        reasons=["В графике"],
        action_items=["Выдать транш"],
        summary="Краткое резюме",
    )
    prompt = build_inspector_prompt(sample_credit, sample_inspection, rec)

    assert "CR-GIGA-001" in prompt
    assert "Сидоров Андрей Владимирович" in prompt
    assert "54:07:055001:178" in prompt
    assert "Кирпич" in prompt
    assert "Зеленый коридор" in prompt


def test_offline_fallback_all_strategies(sample_credit, sample_inspection):
    """Verify offline fallback generates non-empty, strategy-specific legal text."""
    service = GigaChatInspectorService(credentials=None)
    assert service.is_online is False

    for strat in BankStrategy:
        rec = StrategyRecommendation(
            strategy=strat,
            decision_code="CODE",
            risk_level=RiskLevel.MEDIUM,
            summary="Базовое резюме",
        )
        summary = service.generate_summary(sample_credit, sample_inspection, rec)
        assert len(summary) > 50
        assert sample_credit.cadastral_number in summary
        assert "ЗАКЛЮЧЕНИЕ ИНСПЕКТОРА" in summary


def test_mocked_gigachat_api_call(sample_credit, sample_inspection):
    """Verify that when GigaChat SDK is available, API response is prioritized."""
    mock_choice = MagicMock()
    mock_choice.message.content = "ОФИЦИАЛЬНОЕ AI-РЕЗЮМЕ ОТ GIGACHAT: Объект соответствует требованиям."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.return_value = mock_response

    service = GigaChatInspectorService(credentials="mock_token")
    service._client = mock_client
    service._is_online = True

    rec = StrategyRecommendation(
        strategy=BankStrategy.TRANCHE_APPROVAL,
        decision_code="CODE",
        risk_level=RiskLevel.LOW,
        summary="Базовое",
    )

    result = service.generate_summary(sample_credit, sample_inspection, rec)
    assert "ОФИЦИАЛЬНОЕ AI-РЕЗЮМЕ ОТ GIGACHAT" in result
    mock_client.chat.assert_called_once()


def test_graceful_fallback_on_api_exception(sample_credit, sample_inspection):
    """Verify that if GigaChat API throws network/auth error, offline generator takes over."""
    mock_client = MagicMock()
    mock_client.chat.side_effect = ConnectionError("Network timeout to Sber API")

    service = GigaChatInspectorService(credentials="mock_token")
    service._client = mock_client
    service._is_online = True

    rec = StrategyRecommendation(
        strategy=BankStrategy.TRANCHE_APPROVAL,
        decision_code="CODE",
        risk_level=RiskLevel.LOW,
        summary="Базовое",
    )

    result = service.generate_summary(sample_credit, sample_inspection, rec)
    # Shouldn't raise, returns offline summary
    assert "ЗАКЛЮЧЕНИЕ ИНСПЕКТОРА" in result
    assert "Автоматически согласован очередной транш" in result
