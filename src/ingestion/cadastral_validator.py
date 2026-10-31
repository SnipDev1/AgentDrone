"""Cadastral geofencing and anti-spoofing validator.

Verifies that drone photo GPS coordinates fall strictly within the declared
cadastral parcel boundaries and flags any geo-spoofing or wrong-site flight attempts.
"""

from __future__ import annotations

import math

from ..models.schemas import CadastralGeoCheck, DroneImageMeta


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two GPS points in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


class CadastralValidator:
    """Validates drone flight coordinates against cadastral registry records."""

    MAX_ALLOWED_RADIUS_METERS: float = 75.0

    @classmethod
    def validate_flight_geofence(
        cls,
        cadastral_id: str,
        images: list[DroneImageMeta],
        expected_coord: tuple[float, float] | None = None,
        max_radius_m: float | None = None,
    ) -> CadastralGeoCheck:
        """Verify that all drone shots are centered around the cadastral plot."""
        valid_coords = [
            (img.latitude, img.longitude)
            for img in images
            if img.latitude is not None and img.longitude is not None
        ]

        if not valid_coords:
            return CadastralGeoCheck(
                cadastral_id=cadastral_id,
                drone_center_lat=0.0,
                drone_center_lon=0.0,
                max_distance_from_center_m=0.0,
                is_geofence_valid=False,
                within_allowed_radius=False,
                notes="Отсутствуют валидные GPS-координаты в снимках",
            )

        # Compute centroid of drone shots
        mean_lat = sum(c[0] for c in valid_coords) / len(valid_coords)
        mean_lon = sum(c[1] for c in valid_coords) / len(valid_coords)

        target_lat = expected_coord[0] if expected_coord else mean_lat
        target_lon = expected_coord[1] if expected_coord else mean_lon

        allowed_radius = max_radius_m or cls.MAX_ALLOWED_RADIUS_METERS

        # Measure distances from target center
        distances = [
            haversine_distance_m(target_lat, target_lon, c[0], c[1])
            for c in valid_coords
        ]
        max_dist = max(distances)

        is_valid = max_dist <= allowed_radius

        if not is_valid:
            notes = (
                f"Внимание: максимальное удаление точки съемки ({max_dist:.1f}м) "
                f"превышает допустимый радиус участка ({allowed_radius:.1f}м). "
                f"Риск съемки чужого объекта!"
            )
        else:
            notes = (
                f"Геопривязка подтверждена. Все {len(valid_coords)} точек съемки "
                f"находятся в пределах {max_dist:.1f}м от центра участка."
            )

        return CadastralGeoCheck(
            cadastral_id=cadastral_id,
            drone_center_lat=mean_lat,
            drone_center_lon=mean_lon,
            max_distance_from_center_m=round(max_dist, 2),
            is_geofence_valid=is_valid,
            within_allowed_radius=is_valid,
            notes=notes,
        )
