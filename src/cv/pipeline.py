"""Unified Computer Vision Pipeline for AgentDrone.

Executes complete analysis of a cadastral plot:
Drone Ingestion -> Geo & Angle Validation -> CV Feature Extraction ->
Readiness 0/50/100% -> Material Classification -> Anti-Fraud & Defect Analysis ->
Structured Inspection Result.
"""

from __future__ import annotations

from pathlib import Path

from ..config import find_dataset_root
from ..ingestion.angle_detector import AngleCompletenessDetector
from ..ingestion.cadastral_validator import CadastralValidator
from ..ingestion.notes_parser import NotesParser
from ..ingestion.ppk_parser import PPKParser
from ..ingestion.telemetry_parser import TelemetryParser
from ..models.schemas import (
    CreditObject,
    CVInspectionResult,
    DroneImageMeta,
    ShotAngleType,
    WallMaterial,
)
from .feature_extractor import FeatureExtractor, ImageFeatures
from .fraud_detector import FraudDetector
from .material_classifier import MaterialClassifier
from .multispectral import MultispectralAnalyzer
from .readiness_classifier import ReadinessClassifier


class CVPipeline:
    """End-to-end automated drone inspection pipeline."""

    @classmethod
    def analyze_plot(
        cls,
        plot_id: str,
        plot_dir: str | Path | None = None,
        credit_object: CreditObject | None = None,
        declared_material: str | None = None,
    ) -> CVInspectionResult:
        """Run complete inspection analysis for a given plot ID."""
        # 1. Resolve directory
        if plot_dir is None:
            dataset_root = find_dataset_root()
            resolved_dir = dataset_root / plot_id
        else:
            resolved_dir = Path(plot_dir)

        if not resolved_dir.exists() or not resolved_dir.is_dir():
            raise FileNotFoundError(f"Plot directory not found: {resolved_dir}")

        # 2. Ingestion & Telemetry
        mrk_files = list(resolved_dir.rglob("*.MRK")) + list(resolved_dir.rglob("*.mrk"))
        mrk_records = TelemetryParser.parse_mrk_file(mrk_files[0]) if mrk_files else {}

        # Scan for drone JPEG files (filter out non-DJI utility images if any)
        raw_jpgs = sorted(
            f for f in resolved_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg")
        )
        drone_jpgs = [f for f in raw_jpgs if "DJI" in f.name.upper()]
        if not drone_jpgs:
            drone_jpgs = raw_jpgs

        shots_meta: list[DroneImageMeta] = []
        flight_date: str | None = None
        for img_path in drone_jpgs:
            meta = TelemetryParser.parse_image_metadata(img_path, mrk_records=mrk_records)
            shots_meta.append(meta)
            if not flight_date and meta.timestamp:
                try:
                    # e.g. "2027:08:07 14:30:00" -> "2027-08-07"
                    clean_ts = meta.timestamp.strip().split()[0].replace(":", "-")
                    if len(clean_ts) == 10 and clean_ts[4] == "-" and clean_ts[7] == "-":
                        flight_date = clean_ts
                except Exception:
                    pass

        if not flight_date and drone_jpgs:
            from ..engine.dynamics_analyzer import extract_date_from_image
            flight_date = extract_date_from_image(drone_jpgs[0])

        # 3. Angle Completeness & Cadastral Geocheck
        completeness = AngleCompletenessDetector.evaluate(shots_meta)
        geocheck = CadastralValidator.validate_flight_geofence(plot_id, shots_meta)

        # 4. Computer Vision Feature Extraction
        features_list: list[ImageFeatures] = []
        for img_path in drone_jpgs:
            feat = FeatureExtractor.extract_from_file(img_path)
            if feat is not None:
                features_list.append(feat)

        # 5. Readiness & Stage Classification
        readiness_pct, stage, stage_title, structural_elements = ReadinessClassifier.evaluate(features_list)

        # 6. Material Classification
        # Prefer facade shots for material classification
        facade_feats = [
            f for f, m in zip(features_list, shots_meta)
            if m.shot_angle in (
                ShotAngleType.FACADE_NORTH,
                ShotAngleType.FACADE_SOUTH,
                ShotAngleType.FACADE_EAST,
                ShotAngleType.FACADE_WEST,
                ShotAngleType.OBLIQUE_OTHER,
            )
        ]
        target_material_feats = facade_feats if facade_feats else features_list
        detected_material, material_conf = MaterialClassifier.classify(target_material_feats)
        all_mat, all_conf = MaterialClassifier.classify(features_list)
        if all_mat == WallMaterial.SANDWICH_PANEL:
            detected_material, material_conf = all_mat, all_conf

        # 7. Fraud and Defects Detection
        target_decl_material = declared_material or (credit_object.declared_wall_material if credit_object else None)
        loan_purpose = credit_object.loan_purpose if credit_object else "ИЖС"

        fraud_flag, fraud_reasons, defects = FraudDetector.analyze_fraud_and_defects(
            detected_material=detected_material,
            material_confidence=material_conf,
            readiness_stage=stage,
            readiness_pct=readiness_pct,
            structural_elements=structural_elements,
            features_list=features_list,
            declared_material=target_decl_material,
            loan_purpose=loan_purpose,
        )

        # Check telemetry anti-spoofing (EXIF vs MRK consistency)
        inconsistent_shots = [
            m for m in shots_meta
            if not getattr(m, "is_telemetry_consistent", True)
            and getattr(m, "telemetry_discrepancy_m", None) is not None
        ]
        if inconsistent_shots:
            fraud_flag = True
            max_disc = max(m.telemetry_discrepancy_m for m in inconsistent_shots)
            disc_km = max_disc / 1000.0
            fraud_reasons.append(
                f"Фальсификация телеметрии (Metadata Spoofing): расхождение между встроенным EXIF "
                f"снимков и внешним MRK-логом составляет {disc_km:.2f} км ({max_disc:.1f}м > 50м). "
                f"Выявлена попытка подмены геоданных (склейка чужих фотографий с навигационным логом)!"
            )

        # Add geofence warning to fraud reasons if coordinates violate parcel
        if not geocheck.is_geofence_valid:
            fraud_flag = True
            fraud_reasons.append(
                f"Нарушение геозоны: съемка произведена на удалении {geocheck.max_distance_from_center_m}м "
                f"от заявленного участка (риск подлога/съемки чужого объекта)."
            )

        # 8. PPK Geodetic Telemetry, Multispectral, and Operator Notes Ingestion
        ppk_summary = PPKParser.parse_plot_ppk(resolved_dir)
        multispectral = MultispectralAnalyzer.analyze_plot(resolved_dir)
        operator_notes = NotesParser.parse_plot_notes(resolved_dir)
        note_frauds, note_defects = NotesParser.detect_note_hazards(operator_notes)

        for nf in note_frauds:
            if nf not in fraud_reasons:
                fraud_reasons.append(nf)
                fraud_flag = True

        for nd in note_defects:
            if nd not in defects:
                defects.append(nd)

        # 9. Preliminary recommendation hint
        if fraud_flag:
            rec_hint = (
                "СТОП-ФРОД: Передать материалы в Службу безопасности Банка для аудита и реагирования."
            )
        elif stage == 100:
            rec_hint = (
                "РЕГИСТРАЦИЯ ЗАЛОГА: Объект завершен. Направить уведомление о регистрации ипотеки."
            )
        elif stage == 50:
            rec_hint = (
                "ВЫДАЧА ТРАНША / РЕСТРУКТУРИЗАЦИЯ: Коробка возведена. Проверить график финансирования."
            )
        else:
            rec_hint = (
                "ПРОВЕРКА СТАРТА: Строительство не начато. Направить запрос заемщику о начале работ."
            )

        return CVInspectionResult(
            cadastral_id=plot_id,
            readiness_pct=readiness_pct,
            readiness_stage=stage,
            stage_title=stage_title,
            structural_elements=structural_elements,
            detected_material=detected_material,
            material_confidence=material_conf,
            completeness=completeness,
            geocheck=geocheck,
            fraud_risk_flag=fraud_flag,
            fraud_reasons=fraud_reasons,
            defects_detected=defects,
            recommendation_hint=rec_hint,
            analyzed_images_count=len(drone_jpgs),
            shots_metadata=shots_meta,
            ppk_summary=ppk_summary,
            multispectral=multispectral,
            operator_notes=operator_notes,
            inspection_date=flight_date,
        )
