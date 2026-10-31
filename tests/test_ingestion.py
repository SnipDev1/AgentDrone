"""Unit tests for drone data ingestion, telemetry parsing, and geofencing."""

from pathlib import Path

import pytest

from src.ingestion.angle_detector import AngleCompletenessDetector
from src.ingestion.cadastral_validator import CadastralValidator
from src.ingestion.telemetry_parser import TelemetryParser
from src.models.schemas import DroneImageMeta, ShotAngleType


def test_mrk_parsing_mock(tmp_path: Path):
    """Verify MRK log parsing with synthetic records."""
    mrk_file = tmp_path / "test.MRK"
    line1 = (
        "1\t452963.28\t[2430]\t -9,N\t -39,E\t 85,V\t"
        "54.81554599,Lat\t83.24071808,Lon\t241.871,Ellh\t"
        "1.244835, 0.944932, 2.019979\t16,Q\n"
    )
    line2 = (
        "2\t452967.64\t[2430]\t -9,N\t -38,E\t 85,V\t"
        "54.81554702,Lat\t83.24071770,Lon\t241.764,Ellh\t"
        "1.244426, 0.944800, 2.021841\t16,Q\n"
    )
    mrk_content = line1 + line2
    mrk_file.write_text(mrk_content, encoding="utf-8")

    records = TelemetryParser.parse_mrk_file(mrk_file)
    assert len(records) == 2
    assert 1 in records
    assert 2 in records
    assert pytest.approx(records[1].latitude, rel=1e-6) == 54.81554599
    assert pytest.approx(records[1].longitude, rel=1e-6) == 83.24071808
    assert records[1].q_flag == 16
    assert records[1].gps_week == 2430


def test_classify_shot_angle_rules():
    """Verify angle classification for facades and nadirs."""
    # Nadir high
    assert TelemetryParser.classify_shot_angle(pitch=-89.0, yaw=0.0, alt_rel=45.0) == ShotAngleType.NADIR_HIGH
    # Nadir low
    assert TelemetryParser.classify_shot_angle(pitch=-85.0, yaw=90.0, alt_rel=18.0) == ShotAngleType.NADIR_LOW
    # Facade North (yaw ~ 0°)
    assert TelemetryParser.classify_shot_angle(pitch=-45.0, yaw=5.0, alt_rel=15.0) == ShotAngleType.FACADE_NORTH
    # Facade East (yaw ~ 90°)
    assert TelemetryParser.classify_shot_angle(pitch=-45.0, yaw=85.0, alt_rel=15.0) == ShotAngleType.FACADE_EAST
    # Facade South (yaw ~ 180°)
    assert TelemetryParser.classify_shot_angle(pitch=-45.0, yaw=178.0, alt_rel=15.0) == ShotAngleType.FACADE_SOUTH
    # Facade West (yaw ~ -90° / 270°)
    assert TelemetryParser.classify_shot_angle(pitch=-45.0, yaw=-92.0, alt_rel=15.0) == ShotAngleType.FACADE_WEST


def test_angle_completeness_detector():
    """Test full vs incomplete angle detection."""
    complete_images = [
        DroneImageMeta(filename="1.jpg", file_path="1.jpg", shot_angle=ShotAngleType.NADIR_HIGH),
        DroneImageMeta(filename="2.jpg", file_path="2.jpg", shot_angle=ShotAngleType.NADIR_LOW),
        DroneImageMeta(filename="3.jpg", file_path="3.jpg", shot_angle=ShotAngleType.FACADE_NORTH),
        DroneImageMeta(filename="4.jpg", file_path="4.jpg", shot_angle=ShotAngleType.FACADE_EAST),
        DroneImageMeta(filename="5.jpg", file_path="5.jpg", shot_angle=ShotAngleType.FACADE_SOUTH),
        DroneImageMeta(filename="6.jpg", file_path="6.jpg", shot_angle=ShotAngleType.FACADE_WEST),
    ]

    report = AngleCompletenessDetector.evaluate(complete_images)
    assert report.is_fully_compliant is True
    assert report.facades_count == 4
    assert report.nadir_count == 2
    assert len(report.missing_angles) == 0

    # Incomplete set (missing West facade and Nadir High)
    incomplete_images = complete_images[1:5]
    inc_report = AngleCompletenessDetector.evaluate(incomplete_images)
    assert inc_report.is_fully_compliant is False
    assert ShotAngleType.FACADE_WEST.value in inc_report.missing_angles
    assert ShotAngleType.NADIR_HIGH.value in inc_report.missing_angles


def test_real_dataset_plot_ingestion():
    """Verify ingestion on a real dataset sample if dataset is present."""
    plot_dir = Path("dataset/24.09.20206/5419163001661")
    if not plot_dir.exists():
        pytest.skip("Dataset folder not found")

    mrk_files = list(plot_dir.glob("*.MRK"))
    assert len(mrk_files) > 0
    records = TelemetryParser.parse_mrk_file(mrk_files[0])
    assert len(records) > 0

    jpgs = list(plot_dir.glob("*.JPG"))
    assert len(jpgs) == 6

    metas = [TelemetryParser.parse_image_metadata(f, records) for f in jpgs]
    assert len(metas) == 6
    for m in metas:
        assert m.latitude is not None
        assert m.longitude is not None
        assert m.gimbal_pitch_deg is not None

    report = AngleCompletenessDetector.evaluate(metas)
    assert report.is_fully_compliant is True
    assert report.facades_count == 4
    assert report.nadir_count == 2


def test_cadastral_validator_geofence():
    """Test geofence validation for normal and spoofed flights."""
    # Plot center: (54.8184, 83.1632)
    normal_shots = [
        DroneImageMeta(filename="1.jpg", file_path="1.jpg", latitude=54.81845, longitude=83.16329),
        DroneImageMeta(filename="2.jpg", file_path="2.jpg", latitude=54.81840, longitude=83.16320),
    ]
    check = CadastralValidator.validate_flight_geofence("5419163001661", normal_shots)
    assert check.is_geofence_valid is True
    assert check.max_distance_from_center_m < 50.0

    # Spoofed shot (point 1 km away)
    spoofed_shots = normal_shots + [
        DroneImageMeta(filename="fake.jpg", file_path="fake.jpg", latitude=54.83000, longitude=83.16320)
    ]
    check_fake = CadastralValidator.validate_flight_geofence("5419163001661", spoofed_shots, max_radius_m=75.0)
    assert check_fake.is_geofence_valid is False
    assert check_fake.max_distance_from_center_m > 500.0
