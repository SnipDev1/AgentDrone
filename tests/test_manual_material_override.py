"""Unit tests for manual wall material override by collateral inspector."""

from __future__ import annotations

import unittest

from src.cv.fraud_detector import FraudDetector
from src.engine.decision_engine import DecisionEngine
from src.models.schemas import (
    CadastralGeoCheck,
    CreditObject,
    CVInspectionResult,
    StructuralElements,
    WallMaterial,
)


class TestManualMaterialOverride(unittest.TestCase):
    """Test suite verifying manual inspector override updates fraud detection and decisions."""

    def setUp(self):
        self.credit = CreditObject(
            cadastral_id="54:19:160202:125",
            borrower_name="Иванов И.И.",
            credit_agreement_number="CR-TEST-001",
            credit_amount_rub=6_000_000.0,
            tranche_number=1,
            days_overdue=0,
            declared_wall_material="Кирпич",
            loan_purpose="ИЖС",
        )

        self.cv = CVInspectionResult(
            cadastral_id="54:19:160202:125",
            readiness_pct=88.0,
            readiness_stage=100,
            stage_title="Стадия 100%",
            structural_elements=StructuralElements(
                has_foundation=True,
                has_walls=True,
                has_roof=True,
                has_windows_doors=True,
                foundation_confidence=0.95,
                walls_confidence=0.92,
                roof_confidence=0.90,
                windows_doors_confidence=0.85,
            ),
            detected_material=WallMaterial.FRAME_PANEL,  # CV mistakenly flagged cheap frame
            material_confidence=0.85,
            geocheck=CadastralGeoCheck(
                cadastral_id="54:19:160202:125",
                drone_center_lat=55.0,
                drone_center_lon=83.0,
                is_correct_plot=True,
            ),
        )

    def test_fraud_cleared_when_inspector_corrects_to_brick(self):
        """When CV detected cheap frame panel, fraud is flagged; correcting to Brick clears fraud."""
        # 1. Automated state: declared Brick vs detected Frame Panel -> Fraud
        fraud, reasons, _ = FraudDetector.analyze_fraud_and_defects(
            detected_material=self.cv.detected_material,
            material_confidence=self.cv.material_confidence,
            readiness_stage=self.cv.readiness_stage,
            readiness_pct=self.cv.readiness_pct,
            structural_elements=self.cv.structural_elements,
            features_list=[],
            declared_material=self.credit.declared_wall_material,
            loan_purpose=self.credit.loan_purpose,
        )
        self.assertTrue(fraud)
        self.assertTrue(any("подмен" in r.lower() or "несоответствие" in r.lower() or "хищени" in r.lower() for r in reasons))

        # 2. Inspector visual verification: manual override to Brick
        self.cv.detected_material = WallMaterial.BRICK
        fraud_cleared, reasons_cleared, _ = FraudDetector.analyze_fraud_and_defects(
            detected_material=self.cv.detected_material,
            material_confidence=1.0,
            readiness_stage=self.cv.readiness_stage,
            readiness_pct=self.cv.readiness_pct,
            structural_elements=self.cv.structural_elements,
            features_list=[],
            declared_material=self.credit.declared_wall_material,
            loan_purpose=self.credit.loan_purpose,
        )
        self.assertFalse(fraud_cleared)
        self.assertEqual(len(reasons_cleared), 0)

        # 3. Decision Engine: strategy switches to registration / tranche approval
        self.cv.fraud_risk_flag = False
        self.cv.fraud_reasons = []
        rec = DecisionEngine.evaluate(self.cv, self.credit)
        self.assertIn("STRAT-03", rec.strategy_code)

    def test_fraud_raised_when_inspector_reveals_frame_substitution(self):
        """When contract says Brick, but inspector overrides to Frame Panel, fraud is raised."""
        self.cv.detected_material = WallMaterial.FRAME_PANEL
        fraud, reasons, _ = FraudDetector.analyze_fraud_and_defects(
            detected_material=self.cv.detected_material,
            material_confidence=1.0,
            readiness_stage=self.cv.readiness_stage,
            readiness_pct=self.cv.readiness_pct,
            structural_elements=self.cv.structural_elements,
            features_list=[],
            declared_material=self.credit.declared_wall_material,
            loan_purpose=self.credit.loan_purpose,
        )
        self.assertTrue(fraud)
        self.cv.fraud_risk_flag = fraud
        self.cv.fraud_reasons = reasons
        rec = DecisionEngine.evaluate(self.cv, self.credit)
        self.assertEqual(rec.strategy_code, "STRAT-04")
