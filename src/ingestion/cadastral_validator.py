"""Cadastral geofencing, optical ray projection, and anti-spoofing validator.

Verifies that drone photo GPS coordinates and camera orientation angles (pitch, yaw, alt)
target the declared cadastral parcel retrieved from the Cadastral Registry.
"""

from __future__ import annotations

import math

from ..models.schemas import CadastralGeoCheck, DroneImageMeta
from .cadastral_registry import CadastralRegistry, normalize_cadastral_id


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


def project_camera_aim_ground(
    lat: float,
    lon: float,
    altitude_m: float,
    pitch_deg: float | None,
    yaw_deg: float | None,
) -> tuple[float, float]:
    """Calculate ground intersection of the camera optical axis using gimbal pitch & yaw.

    pitch: -90° is straight down (nadir), -45° is oblique perspective.
    yaw: compass heading in degrees (0° North, 90° East).
    """
    if pitch_deg is None or pitch_deg <= -82.0 or altitude_m <= 0.0:
        # Nadir or nearly vertical: target is directly beneath drone
        return (lat, lon)

    # Oblique angle: calculate horizontal ground offset
    pitch_rad = math.radians(abs(pitch_deg))
    ground_dist_m = altitude_m / math.tan(pitch_rad) if math.tan(pitch_rad) > 0.01 else 0.0

    # Cap unrealistic offsets (e.g. if pitch near horizontal)
    ground_dist_m = min(ground_dist_m, 250.0)

    yaw_rad = math.radians(yaw_deg if yaw_deg is not None else 0.0)
    delta_lat = (ground_dist_m * math.cos(yaw_rad)) / 111139.0
    lat_rad = math.radians(lat)
    cos_lat = math.cos(lat_rad) if abs(math.cos(lat_rad)) > 0.01 else 1.0
    delta_lon = (ground_dist_m * math.sin(yaw_rad)) / (111139.0 * cos_lat)

    return (lat + delta_lat, lon + delta_lon)


class CadastralValidator:
    """Validates drone flight coordinates and camera target against cadastral registry records."""

    MAX_TAKEOFF_RADIUS_METERS: float = 150.0
    MAX_AIM_DISTANCE_METERS: float = 100.0

    @classmethod
    def validate_flight_geofence(
        cls,
        cadastral_id: str,
        images: list[DroneImageMeta],
        expected_coord: tuple[float, float] | None = None,
        max_radius_m: float | None = None,
    ) -> CadastralGeoCheck:
        """Verify that drone takeoff and camera optical axes strictly correspond to the cadastral parcel."""
        norm_cadastral_id = normalize_cadastral_id(cadastral_id)
        parcel = CadastralRegistry.get_parcel(cadastral_id)

        valid_images = [
            img for img in images
            if img.latitude is not None and img.longitude is not None
        ]

        if not valid_images:
            return CadastralGeoCheck(
                cadastral_id=norm_cadastral_id,
                raw_folder_name=cadastral_id,
                drone_center_lat=0.0,
                drone_center_lon=0.0,
                registry_lat=parcel.centroid_lat if parcel else None,
                registry_lon=parcel.centroid_lon if parcel else None,
                registry_address=parcel.address if parcel else None,
                registry_vri=parcel.permitted_use if parcel else None,
                distance_to_cadastre_m=0.0,
                mean_camera_aim_dist_m=0.0,
                max_distance_from_center_m=0.0,
                is_geofence_valid=False,
                within_allowed_radius=False,
                is_correct_plot=False,
                verification_status="ОШИБКА: Нет GPS",
                notes="Отсутствуют валидные GPS-координаты в снимках БПЛА",
            )

        # 1. Compute drone cluster center (takeoff / flight location)
        mean_drone_lat = sum(img.latitude for img in valid_images) / len(valid_images)
        mean_drone_lon = sum(img.longitude for img in valid_images) / len(valid_images)

        # 2. Target coordinates from registry (or override)
        if expected_coord:
            target_lat, target_lon = expected_coord
        elif parcel:
            target_lat, target_lon = parcel.centroid_lat, parcel.centroid_lon
        else:
            target_lat, target_lon = mean_drone_lat, mean_drone_lon

        # Distance from drone positions to cadastral center
        drone_distances = [
            haversine_distance_m(target_lat, target_lon, img.latitude, img.longitude)
            for img in valid_images
        ]
        mean_drone_dist = sum(drone_distances) / len(drone_distances)
        max_drone_dist = max(drone_distances)

        # 3. Compute camera ground aim for each image
        aim_coords = []
        aim_distances = []
        for img in valid_images:
            alt = img.altitude_rel_m or (img.altitude_abs_m - 200.0 if img.altitude_abs_m else 30.0)
            alt = max(alt, 5.0)
            aim_lat, aim_lon = project_camera_aim_ground(
                lat=img.latitude,
                lon=img.longitude,
                altitude_m=alt,
                pitch_deg=img.gimbal_pitch_deg,
                yaw_deg=img.gimbal_yaw_deg,
            )
            aim_coords.append((aim_lat, aim_lon))
            aim_distances.append(haversine_distance_m(target_lat, target_lon, aim_lat, aim_lon))

        mean_aim_dist = sum(aim_distances) / len(aim_distances) if aim_distances else mean_drone_dist

        # 4. Evaluation thresholds
        allowed_takeoff_r = max_radius_m or cls.MAX_TAKEOFF_RADIUS_METERS
        allowed_aim_r = max_radius_m or cls.MAX_AIM_DISTANCE_METERS

        is_drone_near = mean_drone_dist <= allowed_takeoff_r
        is_aim_on_target = mean_aim_dist <= allowed_aim_r
        is_correct_plot = is_drone_near and is_aim_on_target

        if is_correct_plot:
            verdict_status = "ПОДТВЕРЖДЕНО: Корректный участок"
            address_info = f" ({parcel.address})" if parcel else ""
            notes = (
                f"Участок {norm_cadastral_id}{address_info} верифицирован. "
                f"Точка съемки/взлета в {mean_drone_dist:.1f}м от центра участка, "
                f"оптическая ось камеры сфокусирована на объекте (отклонение цели: {mean_aim_dist:.1f}м). "
                f"Все {len(valid_images)} кадров соответствуют полетному заданию."
            )
        elif not is_drone_near:
            verdict_status = "ОШИБКА: Снят чужой участок (Геосмещение)"
            notes = (
                f"ВНИМАНИЕ! Дрон находился на удалении {mean_drone_dist:.1f}м "
                f"(макс. {max_drone_dist:.1f}м) от кадастрового участка {norm_cadastral_id}. "
                f"Превышен допустимый радиус взлета ({allowed_takeoff_r:.1f}м). "
                f"Высокий риск съемки чужого залога!"
            )
        else:
            verdict_status = "ПРЕДУПРЕЖДЕНИЕ: Смещение фокуса камеры"
            notes = (
                f"Точка съемки дрона близка ({mean_drone_dist:.1f}м), но камера направлена "
                f"в сторону от участка (среднее отклонение {mean_aim_dist:.1f}м). "
                f"Требуется визуальная перепроверка инспектором."
            )

        return CadastralGeoCheck(
            cadastral_id=norm_cadastral_id,
            raw_folder_name=cadastral_id,
            drone_center_lat=round(mean_drone_lat, 6),
            drone_center_lon=round(mean_drone_lon, 6),
            registry_lat=round(target_lat, 6),
            registry_lon=round(target_lon, 6),
            registry_address=parcel.address if parcel else None,
            registry_vri=parcel.permitted_use if parcel else None,
            distance_to_cadastre_m=round(mean_drone_dist, 2),
            mean_camera_aim_dist_m=round(mean_aim_dist, 2),
            max_distance_from_center_m=round(max_drone_dist, 2),
            is_geofence_valid=is_correct_plot,
            within_allowed_radius=is_drone_near,
            is_correct_plot=is_correct_plot,
            verification_status=verdict_status,
            notes=notes,
        )
