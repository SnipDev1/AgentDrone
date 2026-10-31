"""Helper for assembling ActReportData from pipeline components.

Combines credit contract data, CV drone inspection findings,
and Decision Engine strategy recommendations into an official Act dataset.
Supports both Banking Domain models and CV Inspection pipeline schemas.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, List, Optional

from ..models.credit import CreditObject
from ..models.enums import BankStrategy, CollateralCondition, RiskLevel, StageCategory, WallMaterial
from ..models.inspection import BuildingElements, DefectItem, InspectionResult
from ..models.strategy import ActReportData, ReportPhotoItem, StrategyRecommendation


def build_act_report_data(
    credit: CreditObject,
    inspection: InspectionResult,
    recommendation: StrategyRecommendation,
    photos: Optional[List[ReportPhotoItem]] = None,
    act_date: Optional[date] = None,
    act_number: Optional[str] = None,
) -> ActReportData:
    """Construct complete ActReportData model ready for rendering."""
    target_date = act_date or inspection.inspection_date
    clean_tail = credit.clean_cadastral[-4:] if len(credit.clean_cadastral) >= 4 else "0001"
    target_act_number = act_number or f"АКТ-СИБ-2026-{clean_tail}"

    # Determine collateral condition: Satisfactory only for Tranche Approval and Pledge Registration
    if recommendation.strategy in (BankStrategy.TRANCHE_APPROVAL, BankStrategy.PLEDGE_REGISTRATION):
        condition = CollateralCondition.SATISFACTORY
    else:
        condition = CollateralCondition.UNSATISFACTORY

    return ActReportData(
        act_number=target_act_number,
        act_date=target_date,
        credit=credit,
        inspection=inspection,
        recommendation=recommendation,
        condition=condition,
        photos=photos or [],
    )


def build_act_from_cv(
    cv_result: Any,
    credit_obj: Any,
    decision: Any,
    gigachat_summary: Optional[str] = None,
    photos: Optional[List[ReportPhotoItem]] = None,
    act_date: Optional[date] = None,
    act_number: Optional[str] = None,
) -> ActReportData:
    """Convert CV pipeline schemas (from UI or ingestion) into official ActReportData."""
    # 1. Map Credit
    cadastral_num = getattr(credit_obj, "cadastral_id", "54:00:000000:0")
    if ":" not in cadastral_num and len(cadastral_num) >= 10:
        # Format cadastral like 54:07:055001:178
        c = cadastral_num
        if len(c) == 13:
            cadastral_num = f"{c[:2]}:{c[2:4]}:{c[4:10]}:{c[10:]}"
        elif len(c) == 14:
            cadastral_num = f"{c[:2]}:{c[2:4]}:{c[4:10]}:{c[10:]}"

    proj_mat_str = getattr(credit_obj, "declared_wall_material", "Кирпич")
    proj_mat_enum = WallMaterial.BRICK
    if "блок" in proj_mat_str.lower() or "газобетон" in proj_mat_str.lower():
        proj_mat_enum = WallMaterial.CONCRETE_BLOCK
    elif "дерев" in proj_mat_str.lower() or "брус" in proj_mat_str.lower():
        proj_mat_enum = WallMaterial.TIMBER
    elif "щит" in proj_mat_str.lower() or "каркас" in proj_mat_str.lower() or "sip" in proj_mat_str.lower():
        proj_mat_enum = WallMaterial.FRAME_PANEL

    credit_amount = float(getattr(credit_obj, "credit_amount_rub", 6000000.0))
    tranche_no = int(getattr(credit_obj, "tranche_number", 1))
    tranches_tot = int(getattr(credit_obj, "total_tranches", 3))
    next_tranche = credit_amount / float(tranches_tot)

    try:
        deadline_raw = getattr(credit_obj, "planned_completion_date", "2027-01-01")
        if isinstance(deadline_raw, str):
            deadline_date = datetime.strptime(deadline_raw, "%Y-%m-%d").date()
        else:
            deadline_date = deadline_raw
    except Exception:
        deadline_date = date(2027, 4, 30)

    credit = CreditObject(
        credit_id=getattr(credit_obj, "credit_agreement_number", "CR-2026-001"),
        cadastral_number=cadastral_num,
        borrower_name=getattr(credit_obj, "borrower_name", "Заемщик"),
        credit_amount=credit_amount,
        current_disbursed=(credit_amount / float(tranches_tot)) * tranche_no,
        tranche_number=tranche_no,
        tranches_total=tranches_tot,
        next_tranche_amount=next_tranche,
        credit_issue_date=date(2025, 6, 1),
        construction_deadline=deadline_date,
        days_overdue=int(getattr(credit_obj, "days_overdue", 0)),
        projected_wall_material=proj_mat_enum,
        address=getattr(credit_obj, "address", "Новосибирская область"),
    )

    # 2. Map Inspection
    readiness_pct = float(getattr(cv_result, "readiness_pct", 0.0))
    elems_obj = getattr(cv_result, "structural_elements", None)
    elements = BuildingElements(
        foundation=getattr(elems_obj, "has_foundation", False),
        walls=getattr(elems_obj, "has_walls", False),
        roof=getattr(elems_obj, "has_roof", False),
        windows_doors=getattr(elems_obj, "has_windows_doors", False),
    )

    det_mat_str = str(getattr(cv_result, "detected_material", "Кирпич"))
    det_mat_enum = WallMaterial.BRICK
    if "блок" in det_mat_str.lower() or "газобетон" in det_mat_str.lower():
        det_mat_enum = WallMaterial.CONCRETE_BLOCK
    elif "дерев" in det_mat_str.lower() or "брус" in det_mat_str.lower():
        det_mat_enum = WallMaterial.TIMBER
    elif "щит" in det_mat_str.lower() or "каркас" in det_mat_str.lower():
        det_mat_enum = WallMaterial.FRAME_PANEL

    geocheck = getattr(cv_result, "geocheck", None)
    geo_valid = getattr(geocheck, "is_correct_plot", True) if geocheck else True

    defects_raw = getattr(cv_result, "defects_detected", [])
    defects = [
        DefectItem(defect_type="Выявленный дефект", severity="MEDIUM", description=d)
        for d in defects_raw
    ]

    inspection = InspectionResult(
        cadastral_number=cadastral_num,
        inspection_date=act_date or date(2026, 9, 24),
        total_images_analyzed=int(getattr(cv_result, "analyzed_images_count", 6)),
        view_angles_complete=True,
        geo_match=geo_valid,
        elements=elements,
        physical_readiness_pct=readiness_pct,
        stage=StageCategory.from_percentage(readiness_pct),
        detected_wall_material=det_mat_enum,
        material_match=proj_mat_enum == det_mat_enum,
        is_commercial_building_risk=getattr(cv_result, "fraud_risk_flag", False),
        defects=defects,
    )

    # 3. Map Recommendation & Summary
    strat_code = getattr(decision, "strategy_code", "STRAT-01")
    if "STRAT-01" in strat_code or "TRANCHE" in strat_code:
        strat_enum = BankStrategy.TRANCHE_APPROVAL
        risk_enum = RiskLevel.LOW
    elif "STRAT-02" in strat_code or "RESTRUCTURE" in strat_code:
        strat_enum = BankStrategy.RESTRUCTURING
        risk_enum = RiskLevel.MEDIUM
    elif "STRAT-03" in strat_code or "REGISTER" in strat_code:
        strat_enum = BankStrategy.PLEDGE_REGISTRATION
        risk_enum = RiskLevel.LOW
    elif "STRAT-04" in strat_code or "FRAUD" in strat_code:
        strat_enum = BankStrategy.FRAUD_ALERT
        risk_enum = RiskLevel.CRITICAL
    else:
        strat_enum = BankStrategy.LEGAL_RECOVERY
        risk_enum = RiskLevel.CRITICAL

    final_summary = gigachat_summary or getattr(decision, "action_required", "Официальное экспертное заключение инспектора.")

    recommendation = StrategyRecommendation(
        strategy=strat_enum,
        decision_code=strat_code,
        risk_level=risk_enum,
        confidence=0.95,
        reasons=getattr(decision, "reasoning", ["Решение принято на основании CV-мониторинга"]),
        action_items=[getattr(decision, "action_required", "Выполнить предписание")],
        summary=final_summary,
    )

    return build_act_report_data(
        credit=credit,
        inspection=inspection,
        recommendation=recommendation,
        photos=photos or [],
        act_date=act_date,
        act_number=act_number,
    )
