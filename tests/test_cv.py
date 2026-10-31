"""Unit tests for Computer Vision pipeline, readiness stages, material classifier, and anti-fraud."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.cv.feature_extractor import FeatureExtractor, ImageFeatures
from src.cv.fraud_detector import FraudDetector
from src.cv.material_classifier import MaterialClassifier
from src.cv.pipeline import CVPipeline
from src.cv.readiness_classifier import ReadinessClassifier
from src.models.schemas import CreditObject, StructuralElements, WallMaterial


def test_feature_extractor_synthetic():
    """Verify feature extraction on synthetic green vs structural image."""
    # 1. Pure green vegetation image
    green_img = np.zeros((720, 960, 3), dtype=np.uint8)
    green_img[:, :] = (34, 139, 34)  # ForestGreen in BGR: (34, 139, 34)
    feat_green = FeatureExtractor.extract_from_image(green_img, "green.jpg")
    assert feat_green.vegetation_ratio > 0.85
    assert not feat_green.has_large_structure

    # 2. Structural image with a gray box (building) and lines
    struct_img = np.zeros((720, 960, 3), dtype=np.uint8)
    struct_img[:, :] = (34, 139, 34)  # Grass background
    # Add building rectangle in center
    cv2.rectangle(struct_img, (300, 200), (660, 520), (180, 180, 180), -1)  # Gray box
    # Add dark windows
    cv2.rectangle(struct_img, (340, 250), (420, 340), (20, 20, 20), -1)
    cv2.rectangle(struct_img, (540, 250), (620, 340), (20, 20, 20), -1)

    feat_struct = FeatureExtractor.extract_from_image(struct_img, "struct.jpg")
    assert feat_struct.has_large_structure is True
    assert feat_struct.structure_area_ratio > 0.10


def test_readiness_classifier_stages():
    """Verify mapping of visual features to 0%, 50%, and 100% stages."""
    # Stage 0: Empty green parcel
    empty_feat = ImageFeatures(
        image_name="empty.jpg",
        width=1000,
        height=1000,
        vegetation_ratio=0.95,
        edge_density=0.01,
        horizontal_lines_count=2,
        vertical_lines_count=1,
        structural_density=0.03,
        dominant_hue=45.0,
        mean_saturation=120.0,
        color_shares={},
        has_large_structure=False,
        structure_area_ratio=0.01,
        has_roof_contour=False,
        window_openings_count=0,
    )
    pct_0, stage_0, title_0, elem_0 = ReadinessClassifier.evaluate([empty_feat])
    assert stage_0 == 0
    assert "0%" in title_0
    assert pct_0 < 25.0
    assert elem_0.has_walls is False
    assert elem_0.has_roof is False

    # Stage 50: Box / walls without roof
    box_feat = ImageFeatures(
        image_name="box.jpg",
        width=1000,
        height=1000,
        vegetation_ratio=0.50,
        edge_density=0.15,
        horizontal_lines_count=35,
        vertical_lines_count=25,
        structural_density=0.60,
        dominant_hue=25.0,
        mean_saturation=60.0,
        color_shares={"gray_block": 0.45},
        has_large_structure=True,
        structure_area_ratio=0.22,
        has_roof_contour=False,
        window_openings_count=0,
    )
    pct_50, stage_50, title_50, elem_50 = ReadinessClassifier.evaluate([box_feat])
    assert stage_50 == 50
    assert "50%" in title_50
    assert 25.0 <= pct_50 < 85.0
    assert elem_50.has_foundation is True
    assert elem_50.has_walls is True
    assert elem_50.has_roof is False

    # Stage 100: Finished building with roof and windows
    complete_feat = ImageFeatures(
        image_name="complete.jpg",
        width=1000,
        height=1000,
        vegetation_ratio=0.35,
        edge_density=0.25,
        horizontal_lines_count=50,
        vertical_lines_count=40,
        structural_density=0.90,
        dominant_hue=10.0,
        mean_saturation=70.0,
        color_shares={"red_brick": 0.45},
        has_large_structure=True,
        structure_area_ratio=0.30,
        has_roof_contour=True,
        window_openings_count=6,
    )
    # 2 shots with roof contour
    pct_100, stage_100, title_100, elem_100 = ReadinessClassifier.evaluate(
        [complete_feat, complete_feat]
    )
    assert stage_100 == 100
    assert "100%" in title_100
    assert pct_100 >= 85.0
    assert elem_100.has_foundation is True
    assert elem_100.has_walls is True
    assert elem_100.has_roof is True


def test_material_classifier():
    """Verify material classification based on color and structure shares."""
    brick_feat = ImageFeatures(
        image_name="brick.jpg",
        width=1000,
        height=1000,
        vegetation_ratio=0.50,
        edge_density=0.20,
        horizontal_lines_count=30,
        vertical_lines_count=30,
        structural_density=0.60,
        dominant_hue=8.0,
        mean_saturation=120.0,
        color_shares={
            "red_brick": 0.40,
            "gray_block": 0.05,
            "wood_timber": 0.05,
            "yellow_osb": 0.05,
            "metal_panel": 0.0,
        },
        has_large_structure=True,
        structure_area_ratio=0.25,
        has_roof_contour=True,
        window_openings_count=2,
    )
    mat, conf = MaterialClassifier.classify([brick_feat])
    assert mat == WallMaterial.BRICK
    assert conf >= 0.60


def test_fraud_detector_material_mismatch():
    """Verify Stop-Fraud trigger on material substitution."""
    elem = StructuralElements(has_foundation=True, has_walls=True, has_roof=False)
    fraud_flag, reasons, defects = FraudDetector.analyze_fraud_and_defects(
        detected_material=WallMaterial.FRAME_PANEL,
        material_confidence=0.85,
        readiness_stage=50,
        readiness_pct=50.0,
        structural_elements=elem,
        features_list=[],
        declared_material="Кирпич",
        loan_purpose="ИЖС",
    )
    assert fraud_flag is True
    assert any("Stop-Fraud" in r or "несоответствие" in r.lower() for r in reasons)
    assert any("консервация" in d.lower() for d in defects)


def test_fraud_detector_berdsk_supermarket():
    """Verify fraud detection for commercial supermarket on residential plot (Berdsk case)."""
    elem = StructuralElements(has_foundation=True, has_walls=True, has_roof=True, has_windows_doors=True)
    fraud_flag, reasons, defects = FraudDetector.analyze_fraud_and_defects(
        detected_material=WallMaterial.SANDWICH_PANEL,
        material_confidence=0.90,
        readiness_stage=100,
        readiness_pct=100.0,
        structural_elements=elem,
        features_list=[],
        declared_material="Кирпич",
        loan_purpose="ИЖС",
        is_berdsk_case=True,
    )
    assert fraud_flag is True
    assert any("супермаркет" in r.lower() or "нецелевое" in r.lower() for r in reasons)
    assert any("ЛЭП" in d for d in defects)


def test_cv_pipeline_real_dataset_sample():
    """Verify end-to-end CV inspection on a real dataset sample."""
    plot_dir = Path("dataset/24.09.20206/5419163001661")
    if not plot_dir.exists():
        pytest.skip("Dataset plot not available")

    credit_obj = CreditObject(
        cadastral_id="5419163001661",
        borrower_name="Иванов Иван Иванович",
        credit_agreement_number="КР-2026-ИЖС-0123",
        credit_amount_rub=6500000.0,
        planned_completion_date="2026-12-01",
        declared_wall_material="Кирпич",
    )

    result = CVPipeline.analyze_plot("5419163001661", plot_dir=plot_dir, credit_object=credit_obj)
    assert result.cadastral_id == "5419163001661"
    assert result.analyzed_images_count == 6
    assert 0.0 <= result.readiness_pct <= 100.0
    assert result.readiness_stage in (0, 50, 100)
    assert result.completeness.is_fully_compliant is True
    assert result.geocheck.is_geofence_valid is True
    assert len(result.shots_metadata) == 6


def test_boundary_overlay():
    """Verify cadastral boundary calculation and visual red line rendering."""
    from PIL import Image

    from src.cv.boundary_overlay import BoundaryOverlay
    from src.ingestion.cadastral_registry import CadastralParcel
    from src.models.schemas import DroneImageMeta

    parcel = CadastralParcel(
        cadastral_id="54:19:130102:1231",
        raw_folder_name="54191301021231",
        address="Новосибирская обл., п. Приобский",
        centroid_lat=55.057010,
        centroid_lon=83.147350,
        area_sqm=1000.0,
        permitted_use="Для ИЖС",
    )

    # 1. Ground polygon test
    corners = BoundaryOverlay.calculate_ground_polygon(parcel)
    assert len(corners) == 4
    for lat, lon in corners:
        assert 54.0 < lat < 56.0
        assert 82.0 < lon < 84.0

    # 2. Camera projection test (Nadir 50m)
    meta = DroneImageMeta(
        filename="DJI_0001.JPG",
        file_path="mock/path.jpg",
        latitude=55.057018,
        longitude=83.147330,
        altitude_rel_m=50.0,
        gimbal_pitch_deg=-90.0,
        gimbal_yaw_deg=-129.0,
    )

    img = Image.new("RGB", (960, 720), color=(100, 160, 100))
    poly_pts = BoundaryOverlay.get_projected_polygon(meta, parcel, img_w=960, img_h=720)
    assert len(poly_pts) == 4

    # 3. Visual overlay test
    overlaid = BoundaryOverlay.apply_boundary_overlay(img, meta, parcel)
    assert overlaid.size == (960, 720)
    assert overlaid.mode == "RGB"


def test_boundary_photogrammetric_projection_and_setback():
    """Verify 3D photogrammetric projection and 3-meter metric setback line."""
    from PIL import Image

    from src.cv.boundary_overlay import BoundaryOverlay
    from src.ingestion.cadastral_registry import CadastralRegistry
    from src.models.schemas import DroneImageMeta

    parcel = CadastralRegistry.get_parcel("54:07:055001:178")
    assert parcel is not None
    assert parcel.polygon_wgs84 is not None
    assert len(parcel.polygon_wgs84) == 4

    # High nadir shot telemetry
    meta_nadir = DroneImageMeta(
        filename="DJI_0002_D.JPG",
        file_path="mock/path.jpg",
        latitude=54.815547,
        longitude=83.240718,
        altitude_rel_m=37.5,
        gimbal_pitch_deg=-90.0,
        gimbal_yaw_deg=77.4,
        gimbal_roll_deg=180.0,
    )

    proj_pts = BoundaryOverlay.project_wgs84_polygon(parcel.polygon_wgs84, meta_nadir, 5280, 3956)
    assert len(proj_pts) == 4
    # All points should be within or near the sensor bounds
    for u, v in proj_pts:
        assert -1000 <= u <= 6500
        assert -1000 <= v <= 5500

    # Metric setback (СП 53.13330)
    setback_wgs84 = BoundaryOverlay.compute_metric_setback_wgs84(parcel.polygon_wgs84, setback_m=3.0)
    assert len(setback_wgs84) == 4
    setback_proj = BoundaryOverlay.project_wgs84_polygon(setback_wgs84, meta_nadir, 5280, 3956)
    assert len(setback_proj) == 4

    # Low altitude fallback (<5m)
    meta_low = DroneImageMeta(
        filename="DJI_0005_D.JPG",
        file_path="mock/path.jpg",
        latitude=54.815420,
        longitude=83.240829,
        altitude_rel_m=1.6,
        gimbal_pitch_deg=-89.9,
    )
    assert BoundaryOverlay.project_wgs84_polygon(parcel.polygon_wgs84, meta_low, 5280, 3956) == []

    img_low = Image.new("RGB", (960, 720), color=(120, 120, 120))
    res_low = BoundaryOverlay.apply_boundary_overlay(img_low, meta_low, parcel)
    assert res_low.size == (960, 720)


def test_boundary_oblique_perspective_with_clipping():
    """Verify facade shot perspective projection with near-plane frustum clipping."""
    from src.cv.boundary_overlay import BoundaryOverlay
    from src.ingestion.cadastral_registry import CadastralRegistry
    from src.models.schemas import DroneImageMeta

    parcel = CadastralRegistry.get_parcel("54:19:130102:1231")
    assert parcel is not None

    # Oblique facade shot looking at parcel from side
    meta_facade = DroneImageMeta(
        filename="DJI_0006_D.JPG",
        file_path="mock/path.jpg",
        latitude=55.056800,
        longitude=83.147100,
        altitude_rel_m=34.8,
        gimbal_pitch_deg=-58.1,
        gimbal_yaw_deg=-39.9,
        gimbal_roll_deg=0.0,
    )

    proj_pts = BoundaryOverlay.project_wgs84_polygon(parcel.polygon_wgs84, meta_facade, 5280, 3956)
    assert len(proj_pts) >= 3


def test_multispectral_ndvi_analysis(tmp_path: Path):
    """Verify NDVI computation and land surface classification from multispectral TIFFs."""
    import numpy as np
    from PIL import Image

    from src.cv.multispectral import MultispectralAnalyzer

    # Create synthetic 16-bit TIFF bands
    # Zone 1: soil / building (low NIR, moderate Red) -> NDVI ~ 0.1
    # Zone 2: dense grass (high NIR, low Red) -> NDVI ~ 0.7
    w, h = 100, 100
    nir_data = np.zeros((h, w), dtype=np.uint16)
    r_data = np.zeros((h, w), dtype=np.uint16)

    nir_data[:50, :] = 12000  # soil
    r_data[:50, :] = 10000

    nir_data[50:, :] = 45000  # dense greenery
    r_data[50:, :] = 8000

    im_nir = Image.fromarray(nir_data)
    im_r = Image.fromarray(r_data)

    im_nir.save(tmp_path / "DJI_0001_MS_NIR.TIF")
    im_r.save(tmp_path / "DJI_0001_MS_R.TIF")

    res = MultispectralAnalyzer.analyze_plot(tmp_path, generate_heatmap=False)
    assert res.has_multispectral is True
    assert 0.10 <= res.mean_ndvi <= 0.60
    assert res.construction_soil_pct >= 40.0
    assert res.vegetation_pct >= 40.0
    assert len(res.bands_available) >= 2
