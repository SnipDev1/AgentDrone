"""Telemetry and metadata parser for DJI drone aerial imagery.

Extracts GNSS RTK coordinates, camera gimbal angles (pitch, yaw, roll),
relative and absolute altitudes, timestamps, and matches with Timestamp.MRK logs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..models.schemas import DroneImageMeta, ShotAngleType
from .cadastral_validator import haversine_distance_m


@dataclass
class MRKRecord:
    """High-precision shutter event record from DJI Timestamp.MRK."""
    sequence: int
    gps_time_sec: float
    gps_week: int
    offset_north_mm: float
    offset_east_mm: float
    offset_vert_mm: float
    latitude: float
    longitude: float
    ellipsoidal_height_m: float
    std_lat_m: float
    std_lon_m: float
    std_hgt_m: float
    q_flag: int


class TelemetryParser:
    """Parser for DJI drone image metadata and RTK logs."""

    @staticmethod
    def parse_mrk_file(mrk_path: str | Path) -> dict[int, MRKRecord]:
        """Parse a DJI Timestamp.MRK file into a mapping of sequence -> MRKRecord."""
        path = Path(mrk_path)
        if not path.exists():
            return {}

        records: dict[int, MRKRecord] = {}
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return {}

        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue

            # Example format:
            # 1  452963.28  [2430]  -9,N  -39,E  85,V  54.81,Lat  83.24,Lon  241.8,Ellh  ...  16,Q
            parts = re.split(r"\t+", line)
            if len(parts) < 10:
                continue

            try:
                seq = int(parts[0].strip())
                gps_time = float(parts[1].strip())
                gps_week_m = re.search(r"\[(\d+)\]", parts[2])
                gps_week = int(gps_week_m.group(1)) if gps_week_m else 0

                n_match = re.search(r"([\d\+\-]+),N", parts[3])
                e_match = re.search(r"([\d\+\-]+),E", parts[4])
                v_match = re.search(r"([\d\+\-]+),V", parts[5])
                lat_match = re.search(r"([\d\.\+\-]+),Lat", parts[6])
                lon_match = re.search(r"([\d\.\+\-]+),Lon", parts[7])
                hgt_match = re.search(r"([\d\.\+\-]+),Ellh", parts[8])

                std_parts = [float(x.strip()) for x in parts[9].split(",") if x.strip()]
                std_lat = std_parts[0] if len(std_parts) > 0 else 0.0
                std_lon = std_parts[1] if len(std_parts) > 1 else 0.0
                std_hgt = std_parts[2] if len(std_parts) > 2 else 0.0

                q_match = re.search(r"(\d+),Q", parts[10]) if len(parts) > 10 else None
                q_flag = int(q_match.group(1)) if q_match else 0

                records[seq] = MRKRecord(
                    sequence=seq,
                    gps_time_sec=gps_time,
                    gps_week=gps_week,
                    offset_north_mm=float(n_match.group(1)) if n_match else 0.0,
                    offset_east_mm=float(e_match.group(1)) if e_match else 0.0,
                    offset_vert_mm=float(v_match.group(1)) if v_match else 0.0,
                    latitude=float(lat_match.group(1)) if lat_match else 0.0,
                    longitude=float(lon_match.group(1)) if lon_match else 0.0,
                    ellipsoidal_height_m=float(hgt_match.group(1)) if hgt_match else 0.0,
                    std_lat_m=std_lat,
                    std_lon_m=std_lon,
                    std_hgt_m=std_hgt,
                    q_flag=q_flag,
                )
            except Exception:
                continue

        return records

    @classmethod
    def extract_image_xmp(cls, img_path: str | Path) -> str:
        """Extract raw XMP metadata string from a JPEG file."""
        path = Path(img_path)
        try:
            with Image.open(path) as img:
                xmp = img.info.get("xmp")
                if isinstance(xmp, bytes):
                    return xmp.decode("utf-8", errors="ignore")
                elif isinstance(xmp, str):
                    return xmp
        except Exception:
            pass

        # Fallback: scan binary header for <x:xmpmeta ... </x:xmpmeta>
        try:
            with open(path, "rb") as f:
                header = f.read(131072)  # Read first 128KB
                start = header.find(b"<x:xmpmeta")
                end = header.find(b"</x:xmpmeta>")
                if start != -1 and end != -1:
                    return header[start:end + 12].decode("utf-8", errors="ignore")
        except Exception:
            pass

        return ""

    @classmethod
    def classify_shot_angle(
        cls, pitch: float | None, yaw: float | None, alt_rel: float | None
    ) -> ShotAngleType:
        """Classify drone shot orientation into standard Bank inspection types."""
        if pitch is None:
            return ShotAngleType.UNKNOWN

        # Pitch in DJI: -90° is looking straight down, 0° is looking at horizon
        if pitch <= -72.0:
            # Nadir shot (pointing down)
            if alt_rel is not None and alt_rel >= 30.0:
                return ShotAngleType.NADIR_HIGH
            elif alt_rel is not None:
                return ShotAngleType.NADIR_LOW
            return ShotAngleType.NADIR_OTHER

        # Oblique perspective facade shot (-20° to -72°)
        if -72.0 < pitch <= -15.0:
            if yaw is None:
                return ShotAngleType.OBLIQUE_OTHER
            norm_yaw = (yaw + 360.0) % 360.0
            if (315.0 <= norm_yaw < 360.0) or (0.0 <= norm_yaw < 45.0):
                return ShotAngleType.FACADE_NORTH
            elif 45.0 <= norm_yaw < 135.0:
                return ShotAngleType.FACADE_EAST
            elif 135.0 <= norm_yaw < 225.0:
                return ShotAngleType.FACADE_SOUTH
            elif 225.0 <= norm_yaw < 315.0:
                return ShotAngleType.FACADE_WEST

        return ShotAngleType.UNKNOWN

    @classmethod
    def parse_image_metadata(
        cls,
        image_path: str | Path,
        mrk_records: dict[int, MRKRecord] | None = None,
    ) -> DroneImageMeta:
        """Parse complete metadata from an image, optionally enriched with MRK data."""
        path = Path(image_path)
        xmp_text = cls.extract_image_xmp(path)

        def _xmp_val(tag: str) -> str | None:
            m = re.search(rf'drone-dji:{tag}="([^"]+)"', xmp_text)
            return m.group(1) if m else None

        # Extract XMP tags
        lat_str = _xmp_val("GpsLatitude")
        lon_str = _xmp_val("GpsLongitude")
        alt_rel_str = _xmp_val("RelativeAltitude")
        alt_abs_str = _xmp_val("AbsoluteAltitude")
        pitch_str = _xmp_val("GimbalPitchDegree")
        yaw_str = _xmp_val("GimbalYawDegree")
        roll_str = _xmp_val("GimbalRollDegree")
        flight_yaw_str = _xmp_val("FlightYawDegree")
        rtk_flag_str = _xmp_val("RtkFlag")
        rtk_status_str = _xmp_val("GpsStatus")
        utc_str = _xmp_val("UTCAtExposure")
        drone_model = _xmp_val("DroneModel")

        lat = float(lat_str) if lat_str else None
        lon = float(lon_str) if lon_str else None
        alt_rel = float(alt_rel_str) if alt_rel_str else None
        alt_abs = float(alt_abs_str) if alt_abs_str else None
        pitch = float(pitch_str) if pitch_str else None
        yaw = float(yaw_str) if yaw_str else None
        roll = float(roll_str) if roll_str else None
        flight_yaw = float(flight_yaw_str) if flight_yaw_str else None
        rtk_flag = int(rtk_flag_str) if rtk_flag_str else None

        # Try to extract sequence index from filename (e.g. DJI_..._0004_D.JPG -> 4)
        seq_match = re.search(r"_(\d{4})_D\.", path.name, re.IGNORECASE)
        shutter_idx = int(seq_match.group(1)) if seq_match else None

        std_lat = None
        std_lon = None
        std_hgt = None
        h_acc = None
        gps_week = None
        gps_time_sec = None
        ant_offset = None

        exif_lat = lat
        exif_lon = lon
        mrk_lat = None
        mrk_lon = None
        telemetry_discrepancy_m = None
        is_telemetry_consistent = True

        # Cross-reference with MRK log if available
        if mrk_records and shutter_idx and shutter_idx in mrk_records:
            mrk = mrk_records[shutter_idx]
            mrk_lat = mrk.latitude
            mrk_lon = mrk.longitude
            lat = mrk.latitude
            lon = mrk.longitude
            alt_abs = mrk.ellipsoidal_height_m
            rtk_flag = mrk.q_flag
            rtk_status = "RTK_FIX" if mrk.q_flag in (1, 16) else f"Q_{mrk.q_flag}"
            std_lat = mrk.std_lat_m
            std_lon = mrk.std_lon_m
            std_hgt = mrk.std_hgt_m
            h_acc = math.hypot(mrk.std_lat_m, mrk.std_lon_m)
            gps_week = mrk.gps_week
            gps_time_sec = mrk.gps_time_sec
            ant_offset = (mrk.offset_north_mm, mrk.offset_east_mm, mrk.offset_vert_mm)

            # Anti-spoofing check: Compare embedded EXIF/XMP GNSS with external MRK RTK coordinates
            if exif_lat is not None and exif_lon is not None:
                telemetry_discrepancy_m = round(
                    haversine_distance_m(exif_lat, exif_lon, mrk_lat, mrk_lon), 2
                )
                # Genuine flights have discrepancy < 2-5m. > 50m indicates metadata spoofing/swap
                if telemetry_discrepancy_m > 50.0:
                    is_telemetry_consistent = False
        else:
            rtk_status = rtk_status_str or ("RTK" if rtk_flag == 16 else "STANDARD_GPS")

        # Fallback to standard EXIF if XMP was missing
        if lat is None or lon is None or utc_str is None:
            try:
                with Image.open(path) as img:
                    exif = img.getexif()
                    if utc_str is None and 306 in exif:  # DateTime
                        utc_str = str(exif[306])
            except Exception:
                pass

        # Classify shot angle
        shot_angle = cls.classify_shot_angle(pitch=pitch, yaw=yaw, alt_rel=alt_rel)

        return DroneImageMeta(
            filename=path.name,
            file_path=str(path.resolve()),
            timestamp=utc_str,
            latitude=lat,
            longitude=lon,
            altitude_rel_m=alt_rel,
            altitude_abs_m=alt_abs,
            gimbal_pitch_deg=pitch,
            gimbal_yaw_deg=yaw,
            gimbal_roll_deg=roll,
            flight_yaw_deg=flight_yaw,
            rtk_flag=rtk_flag,
            rtk_status=rtk_status,
            shot_angle=shot_angle,
            shutter_index=shutter_idx,
            drone_model=drone_model or "DJI M3M",
            is_dji=bool(xmp_text or "DJI" in path.name),
            std_lat_m=std_lat,
            std_lon_m=std_lon,
            std_hgt_m=std_hgt,
            horizontal_accuracy_m=round(h_acc, 3) if h_acc is not None else None,
            gps_week=gps_week,
            gps_time_sec=gps_time_sec,
            antenna_offset_mm=ant_offset,
            exif_latitude=exif_lat,
            exif_longitude=exif_lon,
            mrk_latitude=mrk_lat,
            mrk_longitude=mrk_lon,
            telemetry_discrepancy_m=telemetry_discrepancy_m,
            is_telemetry_consistent=is_telemetry_consistent,
        )
