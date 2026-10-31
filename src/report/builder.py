"""Helper for assembling ActReportData from pipeline components.

Combines credit contract data, CV drone inspection findings,
and Decision Engine strategy recommendations into an official Act dataset.
"""

from datetime import date
from typing import List, Optional

from ..models import (
    CreditObject,
    InspectionResult,
    StrategyRecommendation,
    ActReportData,
    ReportPhotoItem,
    CollateralCondition,
    BankStrategy,
)


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
