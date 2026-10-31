"""Unit tests for AgentDrone Pydantic models.

Tests verification of schemas: CreditObject, DroneImageMeta, InspectionResult,
StrategyRecommendation, ActReportData, and domain enums.
"""

from datetime import date, datetime
import pytest
from pydantic import ValidationError

from src.models import (
    StageCategory,
    WallMaterial,
    BankStrategy,
    RiskLevel,
    CameraViewType,
    CollateralCondition,
    CreditObject,
    DroneImageMeta,
    BuildingElements,
    DefectItem,
    InspectionResult,
    StrategyRecommendation,
    ReportPhotoItem,
    ActReportData,
)


def test_stage_category_mapping():
    """Verify continuous percentage mapping to bank normative 3-tier stages."""
    assert StageCategory.from_percentage(0.0) == StageCategory.STAGE_0
    assert StageCategory.from_percentage(15.0) == StageCategory.STAGE_0
    assert StageCategory.from_percentage(20.0) == StageCategory.STAGE_50
    assert StageCategory.from_percentage(55.0) == StageCategory.STAGE_50
    assert StageCategory.from_percentage(84.9) == StageCategory.STAGE_50
    assert StageCategory.from_percentage(85.0) == StageCategory.STAGE_100
    assert StageCategory.from_percentage(100.0) == StageCategory.STAGE_100


def test_credit_object_valid():
    """Test valid CreditObject creation and clean cadastral property."""
    obj = CreditObject(
        credit_id="CR-2026-001",
        cadastral_number="54:07:055001:178",
        borrower_name="Иванов Иван Иванович",
        credit_amount=6000000.0,
        current_disbursed=2000000.0,
        tranche_number=2,
        tranches_total=3,
        next_tranche_amount=2000000.0,
        credit_issue_date=date(2026, 1, 15),
        construction_deadline=date(2027, 1, 15),
        days_overdue=0,
        projected_wall_material=WallMaterial.BRICK,
        projected_floors=2,
        planned_area_sqm=145.5,
        address="Новосибирская обл., пос. Колывань",
        centroid_lat=55.302,
        centroid_lon=82.742,
    )

    assert obj.clean_cadastral == "5407055001178"
    assert obj.credit_amount == 6000000.0
    assert obj.days_overdue == 0
    assert obj.projected_wall_material == WallMaterial.BRICK


def test_credit_object_validation_errors():
    """Test validation errors for invalid cadastral number or negative amount."""
    with pytest.raises(ValidationError):
        CreditObject(
            credit_id="CR-BAD",
            cadastral_number="abc:invalid",  # Not valid digits
            borrower_name="Петров П.П.",
            credit_amount=5000000.0,
            credit_issue_date=date(2026, 1, 1),
            construction_deadline=date(2027, 1, 1),
        )

    with pytest.raises(ValidationError):
        CreditObject(
            credit_id="CR-BAD",
            cadastral_number="54:07:055001:178",
            borrower_name="Петров П.П.",
            credit_amount=-1000.0,  # Negative amount
            credit_issue_date=date(2026, 1, 1),
            construction_deadline=date(2027, 1, 1),
        )


def test_drone_image_meta():
    """Test DroneImageMeta instantiation with RTK/EXIF fields."""
    meta = DroneImageMeta(
        filename="DJI_0001_D.JPG",
        filepath="dataset/5407055001178/DJI_0001_D.JPG",
        timestamp=datetime(2026, 9, 24, 11, 45, 12),
        lat=55.30211,
        lon=82.74245,
        alt_ellipsoid=142.5,
        gimbal_pitch=-45.0,
        gimbal_yaw=180.0,
        view_type=CameraViewType.FACADE,
        rtk_status="50",
        std_lat=0.015,
        std_lon=0.012,
        std_alt=0.025,
    )
    assert meta.filename == "DJI_0001_D.JPG"
    assert meta.gimbal_pitch == -45.0
    assert meta.view_type == CameraViewType.FACADE


def test_inspection_result():
    """Test InspectionResult schema and building elements."""
    elements = BuildingElements(
        foundation=True,
        walls=True,
        roof=True,
        windows_doors=False,
        scaffolding_or_cranes=True,
    )
    defect = DefectItem(
        defect_type="Отсутствие остекления",
        severity="LOW",
        description="Оконные проемы закрыты пленкой",
    )
    res = InspectionResult(
        cadastral_number="5407055001178",
        inspection_date=date(2026, 9, 24),
        total_images_analyzed=6,
        view_angles_complete=True,
        geo_match=True,
        distance_to_boundary_m=3.8,
        boundary_violation=False,
        elements=elements,
        physical_readiness_pct=65.0,
        stage=StageCategory.STAGE_50,
        detected_wall_material=WallMaterial.BRICK,
        material_match=True,
        defects=[defect],
    )
    assert res.elements.walls is True
    assert res.stage == StageCategory.STAGE_50
    assert len(res.defects) == 1
    assert res.defects[0].defect_type == "Отсутствие остекления"


def test_strategy_recommendation_and_act():
    """Test StrategyRecommendation and full ActReportData."""
    credit = CreditObject(
        credit_id="CR-2026-001",
        cadastral_number="5407055001178",
        borrower_name="Сидоров С.С.",
        credit_amount=5000000.0,
        credit_issue_date=date(2026, 1, 1),
        construction_deadline=date(2027, 1, 1),
    )
    inspection = InspectionResult(
        cadastral_number="5407055001178",
        inspection_date=date(2026, 9, 24),
        stage=StageCategory.STAGE_50,
    )
    rec = StrategyRecommendation(
        strategy=BankStrategy.TRANCHE_APPROVAL,
        decision_code="STRAT_TRANCHE_OK",
        risk_level=RiskLevel.LOW,
        confidence=0.98,
        is_automated=True,
        reasons=["График соблюдается", "Коробка возведена"],
        action_items=["Согласовать выплату транша №2"],
        summary="Стройка ведется ритмично, рекомендуется перевод средств.",
    )
    act = ActReportData(
        act_number="АКТ-54-001",
        act_date=date(2026, 9, 25),
        credit=credit,
        inspection=inspection,
        recommendation=rec,
        condition=CollateralCondition.SATISFACTORY,
        photos=[
            ReportPhotoItem(
                file_path="dataset/5407055001178/DJI_0001_D.JPG",
                caption="Северный фасад строения (угол 45°)",
                is_key_view=True,
            )
        ],
    )
    assert act.act_number == "АКТ-54-001"
    assert act.condition == CollateralCondition.SATISFACTORY
    assert act.recommendation.strategy == BankStrategy.TRANCHE_APPROVAL
    assert len(act.photos) == 1
