"""Construction readiness classifier adhering to Bank standard stages.

Computes physical completion percentage (0..100%) based on structural component weights:
Foundation (15%) + Walls (40%) + Roof (30%) + Windows/Doors (15%) -> Discrete 0% / 50% / 100%.
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

        # Check if entire area is empty greenery / raw vegetation with no structures
        avg_veg = sum(f.vegetation_ratio for f in features_list) / len(features_list)
        any_structure = any(f.has_large_structure for f in features_list)
        max_struct_ratio = max(f.structure_area_ratio for f in features_list)
        max_h_lines = max(f.horizontal_lines_count for f in features_list)
        max_v_lines = max(f.vertical_lines_count for f in features_list)
        roof_votes = sum(1 for f in features_list if f.has_roof_contour)
        total_windows = sum(f.window_openings_count for f in features_list)

        # 1. Foundation assessment
        # Foundation exists if there is non-vegetation structural area, ground excavation or slab outline
        has_foundation = False
        foundation_conf = 0.0
        if any_structure or max_struct_ratio >= 0.05 or max_h_lines > 15:
            has_foundation = True
            foundation_conf = min(1.0, 0.4 + (max_struct_ratio * 3.0) + (max_h_lines / 100.0))
        elif avg_veg < 0.75:
            has_foundation = True
            foundation_conf = 0.60
        else:
            has_foundation = False
            foundation_conf = 0.10

        # 2. Walls / Box assessment
        # Walls require vertical lines, significant edge density, and substantial height structure
        has_walls = False
        walls_conf = 0.0
        if has_foundation and (max_v_lines >= 12 or (max_struct_ratio >= 0.08 and max_h_lines >= 25)):
            has_walls = True
            walls_conf = min(1.0, 0.45 + (max_v_lines / 80.0) + (max_struct_ratio * 2.0))
        else:
            has_walls = False
            walls_conf = 0.15

        # 3. Roof assessment
        # Roof requires geometric roof contour or high roof votes across multiple perspectives
        has_roof = False
        roof_conf = 0.0
        if has_walls and (roof_votes >= 2 or (roof_votes >= 1 and max_h_lines >= 40)):
            has_roof = True
            roof_conf = min(1.0, 0.50 + (roof_votes / float(len(features_list))) * 0.5)
        else:
            has_roof = False
            roof_conf = 0.10

        # 4. Windows & Doors / Enclosed thermal contour
        has_windows_doors = False
        win_conf = 0.0
        if has_walls and has_roof and (total_windows >= 3 or max_v_lines >= 35):
            has_windows_doors = True
            win_conf = min(1.0, 0.50 + (total_windows * 0.08))
        else:
            has_windows_doors = False
            win_conf = 0.10

        # Compute continuous readiness percentage (0..100%)
        readiness_pct = (
            (cls.WEIGHT_FOUNDATION if has_foundation else 0.0) * (foundation_conf if has_foundation else 0.0)
            + (cls.WEIGHT_WALLS if has_walls else 0.0) * (walls_conf if has_walls else 0.0)
            + (cls.WEIGHT_ROOF if has_roof else 0.0) * (roof_conf if has_roof else 0.0)
            + (cls.WEIGHT_WINDOWS_DOORS if has_windows_doors else 0.0) * (win_conf if has_windows_doors else 0.0)
        ) * 100.0

        # Calibration based on confirmed elements
        if has_foundation and has_walls and has_roof and has_windows_doors:
            readiness_pct = max(readiness_pct, 88.0)
        elif has_foundation and has_walls and not has_roof:
            readiness_pct = max(35.0, min(readiness_pct, 65.0))
        elif not has_foundation and not has_walls:
            readiness_pct = min(readiness_pct, 12.0)

        readiness_pct = round(max(0.0, min(100.0, readiness_pct)), 1)

        # Map to Bank discrete 3-stage scale (0% / 50% / 100%)
        if readiness_pct < 25.0:
            stage = 0
            stage_title = "0% — Земельный участок (Строительство не начато)"
        elif readiness_pct < 85.0:
            stage = 50
            stage_title = "50% — Коробка здания (Возведены фундамент и стены)"
        else:
            stage = 100
            stage_title = "100% — Завершенное строение (Закрыт контур, готово к регистрации залога)"

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
