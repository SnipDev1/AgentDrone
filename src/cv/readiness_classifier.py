"""Construction readiness classifier adhering to Bank standard stages.

Computes physical completion percentage (0..100%) based on structural component weights:
Foundation (15%) + Walls (40%) + Roof (30%) + Windows/Doors (15%) -> Discrete 0% / 50% / 100%.

Enforces strict domain requirements:
- Empty land plots with grass/soil are strictly 0%.
- Unfinished boxes without window glazing or exterior finishing are capped at 50% stage.
- 100% stage strictly requires an enclosed thermal contour (roof + glazed windows + doors).
"""

from __future__ import annotations

from ..models.schemas import StructuralElements
from .feature_extractor import ImageFeatures


class ReadinessClassifier:
    """Classifies building stage into 0%, 50%, or 100% based on visual features."""

    # Structural component weights according to Bank / construction standards
    WEIGHT_FOUNDATION = 0.15
    WEIGHT_WALLS = 0.40
    WEIGHT_ROOF = 0.30
    WEIGHT_WINDOWS_DOORS = 0.15

    @classmethod
    def evaluate(cls, features_list: list[ImageFeatures]) -> tuple[float, int, str, StructuralElements]:
        """Aggregate image features and determine physical progress and discrete stage."""
        if not features_list:
            return 0.0, 0, "0% — Земельный участок (Строительство не начато)", StructuralElements()

        # Check if parcel is completely empty (no structures inside parcel boundary)
        any_structure = any(f.has_large_structure for f in features_list)
        max_struct_ratio = max(f.structure_area_ratio for f in features_list)
        max_h_lines = max(f.horizontal_lines_count for f in features_list)
        max_v_lines = max(f.vertical_lines_count for f in features_list)
        roof_votes = sum(1 for f in features_list if f.has_roof_contour)
        total_windows = sum(f.window_openings_count for f in features_list)
        total_glazed = sum(getattr(f, "glazed_windows_count", 0) for f in features_list)
        total_empty_openings = sum(getattr(f, "empty_openings_count", 0) for f in features_list)
        finished_roof_votes = sum(1 for f in features_list if getattr(f, "has_finished_roof", False))
        finished_facade_votes = sum(1 for f in features_list if getattr(f, "has_finished_facade", False))
        all_parcel_empty = all(getattr(f, "is_parcel_empty", False) for f in features_list)

        # 0. Early exit for completely vacant land plots
        if all_parcel_empty or (not any_structure and max_struct_ratio < 0.035 and max_v_lines < 8):
            return 0.0, 0, "0% — Земельный участок (Строительство не начато)", StructuralElements(
                has_foundation=False,
                has_walls=False,
                has_roof=False,
                has_windows_doors=False,
                foundation_confidence=0.0,
                walls_confidence=0.0,
                roof_confidence=0.0,
                windows_doors_confidence=0.0,
            )

        # 1. Foundation assessment
        # Foundation requires a confirmed building contour or excavation/slab outline inside parcel
        has_foundation = False
        foundation_conf = 0.0
        if any_structure or max_struct_ratio >= 0.05:
            has_foundation = True
            foundation_conf = min(1.0, 0.50 + (max_struct_ratio * 3.0))
        elif max_h_lines >= 25 and max_struct_ratio >= 0.03:
            has_foundation = True
            foundation_conf = 0.55
        else:
            has_foundation = False
            foundation_conf = 0.05

        # 2. Walls / Box assessment
        # Walls require foundation, vertical lines, and substantial elevation structure
        has_walls = False
        walls_conf = 0.0
        if has_foundation and (max_v_lines >= 12 and (max_h_lines >= 18 or max_struct_ratio >= 0.06)):
            has_walls = True
            walls_conf = min(1.0, 0.45 + (max_v_lines / 60.0) + (max_struct_ratio * 2.0))
        else:
            has_walls = False
            walls_conf = 0.10

        # 3. Roof assessment
        # Roof requires geometric roof contour across multiple perspectives (at least 2 votes or finished roofing)
        has_roof = False
        roof_conf = 0.0
        min_roof_votes = 2 if len(features_list) >= 3 else 1
        if has_walls and roof_votes >= min_roof_votes and (finished_roof_votes >= 1 or max_h_lines >= 35):
            has_roof = True
            roof_conf = min(1.0, 0.50 + (roof_votes / float(len(features_list))) * 0.5)
        else:
            has_roof = False
            roof_conf = 0.10

        # 4. Windows & Doors / Enclosed thermal contour
        # STRICT REQUIREMENT: Must have confirmed glazed windows and closed openings.
        # No more false triggers from vertical lines!
        has_windows_doors = False
        win_conf = 0.0
        if has_walls and has_roof:
            # Glazed windows must be present and not dominated by empty unfinished openings
            if total_glazed >= 2 and (total_glazed >= total_empty_openings or total_empty_openings <= 1):
                has_windows_doors = True
                win_conf = min(1.0, 0.60 + (total_glazed * 0.08))
            elif total_windows >= 4 and total_empty_openings == 0:
                has_windows_doors = True
                win_conf = 0.65
            else:
                has_windows_doors = False
                win_conf = 0.10
        else:
            has_windows_doors = False
            win_conf = 0.05

        # Compute continuous physical readiness percentage (0..100%)
        readiness_pct = (
            (cls.WEIGHT_FOUNDATION if has_foundation else 0.0) * (foundation_conf if has_foundation else 0.0)
            + (cls.WEIGHT_WALLS if has_walls else 0.0) * (walls_conf if has_walls else 0.0)
            + (cls.WEIGHT_ROOF if has_roof else 0.0) * (roof_conf if has_roof else 0.0)
            + (cls.WEIGHT_WINDOWS_DOORS if has_windows_doors else 0.0) * (win_conf if has_windows_doors else 0.0)
        ) * 100.0

        # Calibration according to Bank standard stages
        if not has_foundation and not has_walls:
            readiness_pct = 0.0
            stage = 0
            stage_title = "0% — Земельный участок (Строительство не начато)"
        elif has_foundation and not has_walls:
            readiness_pct = max(8.0, min(readiness_pct, 18.0))
            stage = 0
            stage_title = "0% — Земельный участок (Начат нулевой цикл / фундамент)"
        elif has_foundation and has_walls and (not has_windows_doors or not has_roof):
            # Thermal contour is open: box without roof or box without windows/doors/finishing
            # Strictly capped between 45% and 65% (Stage 50)
            readiness_pct = max(45.0, min(readiness_pct, 65.0))
            stage = 50
            stage_title = "50% — Коробка здания (Возведены фундамент и стены, контур не закрыт)"
        elif has_foundation and has_walls and has_roof and has_windows_doors:
            # Enclosed thermal contour completed
            if finished_facade_votes >= 1:
                readiness_pct = max(readiness_pct, 92.0)
            else:
                readiness_pct = max(readiness_pct, 86.0)
            readiness_pct = min(100.0, readiness_pct)
            stage = 100
            stage_title = "100% — Завершенное строение (Закрыт контур, готово к регистрации залога)"
        else:
            readiness_pct = min(readiness_pct, 20.0)
            stage = 0
            stage_title = "0% — Земельный участок (Строительство не начато)"

        readiness_pct = round(max(0.0, min(100.0, readiness_pct)), 1)

        elements = StructuralElements(
            has_foundation=has_foundation,
            has_walls=has_walls,
            has_roof=has_roof,
            has_windows_doors=has_windows_doors,
            foundation_confidence=round(foundation_conf, 2),
            walls_confidence=round(walls_conf, 2),
            roof_confidence=round(roof_conf, 2),
            windows_doors_confidence=round(win_conf, 2),
        )

        return readiness_pct, stage, stage_title, elements
