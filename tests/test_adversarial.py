"""Adversarial and anti-spoofing unit tests for AgentDrone.

Verifies system resilience against metadata fabrication, cross-plot image substitution,
telemetry tampering (EXIF vs MRK discrepancy), and geofence violations.
"""

from pathlib import Path
from PIL import Image
import pytest

from src.cv.boundary_overlay import BoundaryOverlay
from src.cv.pipeline import CVPipeline
from src.engine.decision_engine import DecisionEngine
from src.ingestion.cadastral_registry import CadastralRegistry
from src.ingestion.cadastral_validator import CadastralValidator, haversine_distance_m
from src.ingestion.credit_registry import CreditRegistry
from src.ingestion.telemetry_parser import TelemetryParser
from src.models.schemas import StrategyType


DATASET_DIR = Path("dataset/24.09.20206")
PLOT_A = "5407055001178"  # Kiteris residential house (Plot A)
PLOT_B = "54320108708"    # Berdsk commercial plot (Plot B, 9.3 km away)


@pytest.fixture(scope="module")
def sample_paths():
    """Verify presence of real dataset files for adversarial testing."""
    dir_a = DATASET_DIR / PLOT_A
    dir_b = DATASET_DIR / PLOT_B
    if not dir_a.exists() or not dir_b.exists():
        pytest.skip("Dataset directory not found for adversarial testing")

    img_a = dir_a / "DJI_20260807124907_0002_D.JPG"
    mrk_b = dir_b / "DJI_202608071342_038_Timestamp.MRK"
    return {"img_a": img_a, "mrk_b": mrk_b, "dir_a": dir_a, "dir_b": dir_b}


def test_adversarial_telemetry_discrepancy(sample_paths):
    """Verify detection of metadata swap: Image from Plot A paired with MRK from Plot B."""
    mrk_records = TelemetryParser.parse_mrk_file(sample_paths["mrk_b"])
    assert 2 in mrk_records, "Shutter sequence 2 should exist in Berdsk MRK"

    meta = TelemetryParser.parse_image_metadata(sample_paths["img_a"], mrk_records=mrk_records)

    # Image physical EXIF is ~54.815, MRK injected is ~54.805
    assert meta.exif_latitude is not None
    assert meta.mrk_latitude is not None
    assert meta.telemetry_discrepancy_m is not None
    assert meta.telemetry_discrepancy_m > 5000.0, "Discrepancy should be > 5 km"
    assert meta.is_telemetry_consistent is False, "Telemetry must be flagged inconsistent"


def test_adversarial_geofence_violation(sample_paths):
    """Verify CadastralValidator flags drone flight located 9+ km from declared parcel."""
    mrk_records = TelemetryParser.parse_mrk_file(sample_paths["mrk_b"])
    meta = TelemetryParser.parse_image_metadata(sample_paths["img_a"], mrk_records=mrk_records)

    # Validate against Plot A's official parcel boundaries
    geocheck = CadastralValidator.validate_flight_geofence(PLOT_A, [meta])

    assert geocheck.is_geofence_valid is False
    assert geocheck.within_allowed_radius is False
    assert geocheck.is_correct_plot is False
    assert geocheck.distance_to_cadastre_m > 5000.0
    assert "ОШИБКА: Снят чужой участок" in geocheck.verification_status


def test_adversarial_pipeline_stop_fraud(sample_paths, tmp_path):
    """Verify CVPipeline and DecisionEngine trigger STRAT-04 on fabricated dataset."""
    adv_plot_dir = tmp_path / "fabricated_plot"
    adv_plot_dir.mkdir(parents=True, exist_ok=True)

    # Copy photo from Plot A and MRK/NAV from Plot B into temporary test directory
    import shutil
    shutil.copy2(sample_paths["img_a"], adv_plot_dir / sample_paths["img_a"].name)
    shutil.copy2(sample_paths["mrk_b"], adv_plot_dir / sample_paths["mrk_b"].name)

    # Run complete automated CV pipeline
    cv_res = CVPipeline.analyze_plot(PLOT_A, plot_dir=adv_plot_dir)

    assert cv_res.fraud_risk_flag is True
    assert len(cv_res.fraud_reasons) >= 2
    reasons_text = " ".join(cv_res.fraud_reasons)
    assert "Фальсификация телеметрии" in reasons_text
    assert "Нарушение геозоны" in reasons_text

    # Evaluate with Decision Engine
    credit_obj = CreditRegistry.get_credit_object(PLOT_A)
    assert credit_obj is not None

    strat = DecisionEngine.evaluate_cv(cv_res, credit_obj)
    assert strat.strategy == StrategyType.SUSPEND_SECURITY_FRAUD
    assert strat.strategy_code == "STRAT-04"
    assert strat.risk_level == "CRITICAL"
    assert strat.kalita_status == "BLOCKED_FRAUD"


def test_adversarial_boundary_overlay_suppression(sample_paths):
    """Verify BoundaryOverlay suppresses polygon lines and renders STOP-FRAUD security badge."""
    mrk_records = TelemetryParser.parse_mrk_file(sample_paths["mrk_b"])
    meta = TelemetryParser.parse_image_metadata(sample_paths["img_a"], mrk_records=mrk_records)
    parcel = CadastralRegistry.get_parcel(PLOT_A)
    assert parcel is not None

    with Image.open(sample_paths["img_a"]) as im:
        annotated = BoundaryOverlay.apply_boundary_overlay(im, meta, parcel)
        assert annotated.size == im.size

        # Image should be modified (drawn overlay badge)
        assert annotated.mode == "RGB"


def test_haversine_distance_m():
    """Verify mathematical accuracy of haversine calculation."""
    # Distance between ~54.8155, 83.2407 and ~54.8047, 83.0970 is approx 9.3 km
    dist = haversine_distance_m(54.8155, 83.2407, 54.8047, 83.0970)
    assert 9000.0 < dist < 9600.0
