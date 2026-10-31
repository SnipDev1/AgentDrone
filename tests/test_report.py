"""Unit tests for Collateral Inspection Act report generator (TB-05, ENG-03).

Verifies Jinja2 rendering of official Appendix 1 (Приложение 1),
photo table generation, risk condition classification, and file export.
"""

from datetime import date
import pytest

from src.models import (
    CreditObject,
    InspectionResult,
    BuildingElements,
    DefectItem,
    StrategyRecommendation,
    ReportPhotoItem,
    BankStrategy,
    RiskLevel,
    StageCategory,
    WallMaterial,
    CollateralCondition,
)
from src.report import ActGenerator, build_act_report_data


@pytest.fixture
def sample_credit():
    return CreditObject(
        credit_id="CR-SIB-2026-001",
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
        projected_floors=2,
        planned_area_sqm=155.0,
        address="Новосибирская обл., р.п. Колывань, ул. Советская, 14",
    )


@pytest.fixture
def sample_inspection():
    return InspectionResult(
        cadastral_number="5407055001178",
        inspection_date=date(2026, 9, 24),
        total_images_analyzed=6,
        view_angles_complete=True,
        geo_match=True,
        distance_to_boundary_m=4.2,
        boundary_violation=False,
        elements=BuildingElements(foundation=True, walls=True, roof=True, windows_doors=False),
        physical_readiness_pct=65.0,
        stage=StageCategory.STAGE_50,
        detected_wall_material=WallMaterial.BRICK,
        material_match=True,
        defects=[
            DefectItem(
                defect_type="Отсутствие консервации оконных проемов",
                severity="LOW",
                description="Оконные проемы не затянуты защитной пленкой перед зимним периодом",
                recommendation="Загерметизировать проемы до 1 ноября 2026 г.",
            )
        ],
    )


def test_builder_satisfactory_condition(sample_credit, sample_inspection):
    """Verify that Tranche Approval generates SATISFACTORY condition."""
    rec = StrategyRecommendation(
        strategy=BankStrategy.TRANCHE_APPROVAL,
        decision_code="TRANCHE_APPROVAL_STP",
        risk_level=RiskLevel.LOW,
        summary="Стройка в графике.",
    )
    act_data = build_act_report_data(sample_credit, sample_inspection, rec)

    assert act_data.condition == CollateralCondition.SATISFACTORY
    assert "1178" in act_data.act_number


def test_builder_unsatisfactory_condition_on_fraud(sample_credit, sample_inspection):
    """Verify that Fraud Alert generates UNSATISFACTORY condition."""
    rec = StrategyRecommendation(
        strategy=BankStrategy.FRAUD_ALERT,
        decision_code="FRAUD_COMMERCIAL_USE",
        risk_level=RiskLevel.CRITICAL,
        summary="Нецелевое использование.",
    )
    act_data = build_act_report_data(sample_credit, sample_inspection, rec)

    assert act_data.condition == CollateralCondition.UNSATISFACTORY


def test_act_generator_render_html(sample_credit, sample_inspection):
    """Verify HTML rendering contains all essential Sber bank elements."""
    rec = StrategyRecommendation(
        strategy=BankStrategy.TRANCHE_APPROVAL,
        decision_code="TRANCHE_APPROVAL_STP",
        risk_level=RiskLevel.LOW,
        summary="Официальное резюме инспектора для АС «Калита».",
        action_items=["Согласовать выплату транша №2"],
    )
    photos = [
        ReportPhotoItem(
            file_path="dataset/5407055001178/DJI_0001_D.JPG",
            caption="Северный фасад (45°)",
            is_key_view=True,
        ),
        ReportPhotoItem(
            file_path="dataset/5407055001178/DJI_0002_D.JPG",
            caption="Надир кровли",
            is_key_view=True,
        ),
    ]

    act_data = build_act_report_data(sample_credit, sample_inspection, rec, photos=photos)
    generator = ActGenerator()
    html = generator.render_html(act_data)

    assert "<!DOCTYPE html>" in html
    assert "АКТ ПРОВЕРКИ ИМУЩЕСТВА" in html
    assert "Приложение № 1" in html
    assert "АС «Калита»" in html
    assert "CR-SIB-2026-001" in html
    assert "Сидоров Андрей Владимирович" in html
    assert "65.0%" in html
    assert "Отсутствие консервации оконных проемов" in html
    assert "Северный фасад" in html
    assert "Удовлетворительное" in html


def test_act_generator_save_html(sample_credit, sample_inspection, tmp_path):
    """Verify saving report to file creates valid file on disk."""
    rec = StrategyRecommendation(
        strategy=BankStrategy.TRANCHE_APPROVAL,
        decision_code="TRANCHE_APPROVAL_STP",
        risk_level=RiskLevel.LOW,
        summary="Тестовое резюме.",
    )
    act_data = build_act_report_data(sample_credit, sample_inspection, rec)
    generator = ActGenerator()

    out_file = tmp_path / "output_acts" / "act_test.html"
    saved_path = generator.save_html(act_data, out_file)

    assert saved_path.exists()
    assert saved_path.stat().st_size > 1000


def test_build_act_from_cv_schemas():
    """Verify build_act_from_cv integrates CV schemas and renders official HTML Act."""
    from src.models.schemas import (
        CVInspectionResult,
        CadastralGeoCheck,
        CreditObject as CVCreditObject,
        StrategyRecommendation as CVStrategyRecommendation,
        StrategyType,
        StructuralElements,
        WallMaterial as CVWallMaterial,
    )
    from src.report import build_act_from_cv, ActGenerator

    cv_res = CVInspectionResult(
        cadastral_id="5407055001178",
        readiness_pct=60.0,
        readiness_stage=50,
        stage_title="Коробка здания",
        structural_elements=StructuralElements(has_foundation=True, has_walls=True),
        detected_material=CVWallMaterial.BRICK,
        geocheck=CadastralGeoCheck(
            cadastral_id="5407055001178",
            drone_center_lat=55.302,
            drone_center_lon=82.742,
            is_correct_plot=True,
        ),
        fraud_risk_flag=False,
    )

    credit = CVCreditObject(
        cadastral_id="5407055001178",
        borrower_name="Сидоров Андрей Владимирович",
        credit_agreement_number="CR-SIB-2026-001",
        credit_amount_rub=6200000.0,
        planned_completion_date="2027-04-30",
        declared_wall_material="Кирпич",
        days_overdue=0,
    )

    decision = CVStrategyRecommendation(
        strategy=StrategyType.APPROVE_TRANCHE,
        strategy_code="STRAT-01",
        title="Выдача очередного транша",
        risk_level="LOW",
        action_required="Выдать транш №2",
        reasoning=["Стройка в графике"],
        financial_impact_rub=2000000.0,
        kalita_status="APPROVED_TRANCHE",
    )

    act_data = build_act_from_cv(
        cv_result=cv_res,
        credit_obj=credit,
        decision=decision,
        gigachat_summary="Экспертное заключение от GigaChat.",
    )

    assert "1178" in act_data.act_number
    assert act_data.credit.borrower_name == "Сидоров Андрей Владимирович"
    assert act_data.recommendation.summary == "Экспертное заключение от GigaChat."

    gen = ActGenerator()
    html = gen.render_html(act_data)
    assert "АКТ ПРОВЕРКИ ИМУЩЕСТВА" in html
    assert "Экспертное заключение от GigaChat." in html
