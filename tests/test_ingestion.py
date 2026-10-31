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


def test_cadastral_registry_and_normalization():
    """Verify folder name normalization and registry parcel lookup."""
    from src.ingestion.cadastral_registry import CadastralRegistry, normalize_cadastral_id

    # Test folder name formatting
    assert normalize_cadastral_id("5407055001178") == "54:07:055001:178"
    assert normalize_cadastral_id("54191301021231") == "54:19:130102:1231"
    assert normalize_cadastral_id("54320108708") == "54:32:010870:8"
    assert normalize_cadastral_id("54:32:010870:8") == "54:32:010870:8"

    # Test registry lookup by raw folder name
    parcel = CadastralRegistry.get_parcel("5407055001178")
    assert parcel is not None
    assert parcel.cadastral_id == "54:07:055001:178"
    assert "Тогучинский" in parcel.district_name
    assert parcel.centroid_lat > 54.0
    assert parcel.centroid_lon > 83.0
    assert "ИЖС" in parcel.permitted_use

    # Test Berdsk supermarket case in registry
    berdsk_parcel = CadastralRegistry.get_parcel("54320108708")
    assert berdsk_parcel is not None
    assert berdsk_parcel.cadastral_id == "54:32:010870:8"
    assert "Энгельса" in berdsk_parcel.address
    assert berdsk_parcel.is_residential_vri is False


def test_optical_camera_aim_projection():
    """Verify ground projection of camera optical axis from altitude and pitch."""
    from src.ingestion.cadastral_validator import project_camera_aim_ground

    drone_lat, drone_lon = 54.8155, 83.2407
    # 1. Nadir (-90°) -> target is directly beneath drone
    target_lat, target_lon = project_camera_aim_ground(
        drone_lat, drone_lon, altitude_m=30.0, pitch_deg=-90.0, yaw_deg=0.0
    )
    assert target_lat == drone_lat
    assert target_lon == drone_lon

    # 2. Oblique (-45°), facing North (yaw = 0°) -> offset North (higher latitude)
    target_lat_n, target_lon_n = project_camera_aim_ground(
        drone_lat, drone_lon, altitude_m=30.0, pitch_deg=-45.0, yaw_deg=0.0
    )
    assert target_lat_n > drone_lat
    assert pytest.approx(target_lon_n, rel=1e-5) == drone_lon

    # 3. Oblique (-45°), facing East (yaw = 90°) -> offset East (higher longitude)
    target_lat_e, target_lon_e = project_camera_aim_ground(
        drone_lat, drone_lon, altitude_m=30.0, pitch_deg=-45.0, yaw_deg=90.0
    )
    assert pytest.approx(target_lat_e, rel=1e-5) == drone_lat
    assert target_lon_e > drone_lon


def test_cadastral_validator_geofence():
    """Test geofence validation for normal and spoofed flights."""
    # Plot center for 5419163001661: (54.818450, 83.163300)
    normal_shots = [
        DroneImageMeta(
            filename="1.jpg",
            file_path="1.jpg",
            latitude=54.81845,
            longitude=83.16329,
            altitude_rel_m=25.0,
            gimbal_pitch_deg=-90.0,
            gimbal_yaw_deg=0.0,
        ),
        DroneImageMeta(
            filename="2.jpg",
            file_path="2.jpg",
            latitude=54.81840,
            longitude=83.16320,
            altitude_rel_m=20.0,
            gimbal_pitch_deg=-45.0,
            gimbal_yaw_deg=45.0,
        ),
    ]
    check = CadastralValidator.validate_flight_geofence("5419163001661", normal_shots)
    assert check.is_geofence_valid is True
    assert check.is_correct_plot is True
    assert check.cadastral_id == "54:19:163001:661"
    assert "ПОДТВЕРЖДЕНО" in check.verification_status
    assert check.distance_to_cadastre_m < 50.0

    # Spoofed shot: drone took off 2 km away
    spoofed_shots = [
        DroneImageMeta(
            filename="fake.jpg",
            file_path="fake.jpg",
            latitude=54.84000,
            longitude=83.16320,
            altitude_rel_m=30.0,
            gimbal_pitch_deg=-90.0,
        )
    ]
    check_fake = CadastralValidator.validate_flight_geofence("5419163001661", spoofed_shots)
    assert check_fake.is_geofence_valid is False
    assert check_fake.is_correct_plot is False
    assert "ОШИБКА" in check_fake.verification_status
    assert check_fake.distance_to_cadastre_m > 1000.0
