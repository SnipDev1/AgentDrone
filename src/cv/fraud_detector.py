"""Anti-fraud and construction defect detector.

Identifies material substitution frauds, commercial misuse of residential (ИЖС) land,
long-term construction abandonment, and lack of weather preservation.
"""

from __future__ import annotations

from ..models.schemas import StructuralElements, WallMaterial
from .feature_extractor import ImageFeatures


class FraudDetector:
    """Detects fraud risks and construction defects from CV and credit metadata."""

    @classmethod
    def analyze_fraud_and_defects(
        cls,
        detected_material: WallMaterial,
        material_confidence: float,
        readiness_stage: int,
        readiness_pct: float,
        structural_elements: StructuralElements,
        features_list: list[ImageFeatures],
        declared_material: str | None = None,
        loan_purpose: str = "ИЖС",
        is_berdsk_case: bool = False,
    ) -> tuple[bool, list[str], list[str]]:
        """Return (fraud_risk_flag, fraud_reasons, defects_detected)."""
        fraud_reasons: list[str] = []
        defects: list[str] = []

        # 1. Non-residential / commercial building on residential plot (general CV rule)
        if detected_material == WallMaterial.SANDWICH_PANEL and "ижс" in loan_purpose.lower():
            fraud_reasons.append(
                "Признаки нецелевого использования ЗУ: нецелевое использование участка под коммерческий объект. "
                "На аэрофотосъемке зафиксировано крупное нежилое/коммерческое строение "
                "(сэндвич-панели / быстровозводимый конструктив торгового/складского типа, супермаркет) "
                "на участке, предназначенном для индивидуального жилищного строительства (ИЖС). "
                "Риск признания самовольной постройкой (ст. 222 ГК РФ) и нарушения ВРИ земельного участка (ст. 42 ЗК РФ)."
            )

        # 2. Material substitution check (Discrepancy with credit estimate)
        if declared_material and detected_material != WallMaterial.UNKNOWN and material_confidence >= 0.60:
            decl_lower = declared_material.lower()

            # Brick declared vs cheap materials
            if "кирпич" in decl_lower:
                if detected_material == WallMaterial.FRAME_PANEL:
                    fraud_reasons.append(
                        f"Критическое несоответствие материалов (Stop-Fraud): по кредитному договору "
                        f"заявлен капитальный «{declared_material}», фактически возведен дешевый "
                        f"«{detected_material.value}». Риск хищения/нецелевого расходования кредитных средств!"
                    )
                elif detected_material == WallMaterial.TIMBER:
                    fraud_reasons.append(
                        f"Несоответствие проектных решений: по договору заявлен «{declared_material}», "
                        f"фактически строится деревянный дом («{detected_material.value}»)."
                    )

            # Gas concrete declared vs cheap frame
            elif ("блок" in decl_lower or "газобетон" in decl_lower) and detected_material == WallMaterial.FRAME_PANEL:
                fraud_reasons.append(
                    f"Несоответствие материалов: заявлен «{declared_material}», "
                    f"фактически возведен щитовой каркас («{detected_material.value}»)."
                )

        # 3. Defect: Lack of preservation for incomplete construction
        if readiness_stage == 50 and not structural_elements.has_roof:
            defects.append(
                "Отсутствует консервация незавершенного строительства: несущие стены открыты "
                "атмосферным осадкам (дождь, снег), риск намокания и разрушения кладки в зимний период."
            )

        # 4. Defect: Weeds / overgrown vegetation indicating long-term stall
        if features_list:
            avg_veg = sum(f.vegetation_ratio for f in features_list) / len(features_list)
            if readiness_stage == 50 and avg_veg > 0.65:
                defects.append(
                    "Признаки приостановки строительства (долгострой): территория вокруг фундамента "
                    "и стен заросла дикорастущей травой и кустарником, строительная техника и рабочие отсутствуют."
                )

        # 5. Defect: Roof without enclosed windows
        if structural_elements.has_roof and not structural_elements.has_windows_doors and readiness_stage < 100:
            defects.append(
                "Тепловой контур не закрыт: оконные и дверные проемы не остеклены, объект не защищен от вандализма."
            )

        fraud_risk_flag = len(fraud_reasons) > 0
        return fraud_risk_flag, fraud_reasons, defects
