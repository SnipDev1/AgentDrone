"""Angle completeness detector for Bank inspection compliance.

Verifies that the drone flight covers all 6 mandatory perspectives:
4 facades (North, South, East, West) + 2 nadirs (low detailed, high overview).
"""

from __future__ import annotations

from typing import ClassVar

from ..models.schemas import CompletenessReport, DroneImageMeta, ShotAngleType


class AngleCompletenessDetector:
    """Evaluates drone image coverage against Bank requirements."""

    REQUIRED_FACADES: ClassVar[set[ShotAngleType]] = {
        ShotAngleType.FACADE_NORTH,
        ShotAngleType.FACADE_SOUTH,
        ShotAngleType.FACADE_EAST,
        ShotAngleType.FACADE_WEST,
    }

    REQUIRED_NADIRS: ClassVar[set[ShotAngleType]] = {
        ShotAngleType.NADIR_LOW,
        ShotAngleType.NADIR_HIGH,
    }

    @classmethod
    def evaluate(cls, images: list[DroneImageMeta]) -> CompletenessReport:
        """Analyze a collection of images and produce a compliance report."""
        present_set: set[ShotAngleType] = set()
        angle_counts: dict[str, int] = {}

        for img in images:
            present_set.add(img.shot_angle)
            angle_counts[img.shot_angle.value] = angle_counts.get(img.shot_angle.value, 0) + 1

        missing: list[str] = []
        warnings: list[str] = []

        # Check facades
        missing_facades = cls.REQUIRED_FACADES - present_set
        for mf in sorted(missing_facades, key=lambda x: x.value):
            missing.append(mf.value)
            warnings.append(f"Не выполнен обязательный ракурс: {mf.value}")

        # Check nadirs
        missing_nadirs = cls.REQUIRED_NADIRS - present_set
        for mn in sorted(missing_nadirs, key=lambda x: x.value):
            missing.append(mn.value)
            warnings.append(f"Не выполнен обязательный ракурс: {mn.value}")

        facades_count = sum(1 for img in images if img.shot_angle in cls.REQUIRED_FACADES)
        nadir_count = sum(
            1 for img in images if img.shot_angle in (
                ShotAngleType.NADIR_LOW,
                ShotAngleType.NADIR_HIGH,
                ShotAngleType.NADIR_OTHER,
            )
        )

        has_4_facades = len(missing_facades) == 0
        has_2_nadir = len(missing_nadirs) == 0
        is_compliant = has_4_facades and has_2_nadir

        if not is_compliant and len(images) > 0:
            warnings.append(
                f"Нарушение регламента съемки БПЛА: зафиксировано {len(images)} кадров, "
                f"но отсутствуют {len(missing)} ключевых ракурсов."
            )

        present_list = [a.value for a in sorted(present_set, key=lambda x: x.value)]

        return CompletenessReport(
            total_shots=len(images),
            facades_count=facades_count,
            nadir_count=nadir_count,
            has_4_facades=has_4_facades,
            has_2_nadir=has_2_nadir,
            is_fully_compliant=is_compliant,
            present_angles=present_list,
            missing_angles=missing,
            warnings=warnings,
        )
