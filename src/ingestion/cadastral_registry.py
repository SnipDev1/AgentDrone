"""Dynamic Cadastral Registry client (ППК Росреестр / НСПД) for AgentDrone.

Fetches official cadastral parcel records (boundaries in WGS-84, centroids, addresses,
area, cadastral value, and VRI / permitted use) directly from the Russian National Spatial
Data System (ФГИС ЕЦП НСПД / ППК Росреестр API) with local JSON caching and automatic
RTK telemetry fallback for newly uploaded parcels when offline.
"""

from __future__ import annotations

import json
import math
import re
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


def normalize_cadastral_id(folder_or_cadastral: str) -> str:
    """Normalize a raw folder name (e.g. '5407055001178') into a standard cadastral ID.

    Format: RR:DD:QQQQQQ:NNNN
    Example: '5407055001178' -> '54:07:055001:178'
             '54320108708'   -> '54:32:010870:8'
    """
    cleaned = re.sub(r"[^\d]", "", folder_or_cadastral)
    if not cleaned or len(cleaned) < 10:
        return folder_or_cadastral

    if ":" in folder_or_cadastral and folder_or_cadastral.count(":") == 3:
        return folder_or_cadastral.strip()

    region = cleaned[:2]
    district = cleaned[2:4]
    quarter = cleaned[4:10]
    parcel = cleaned[10:]
    return f"{region}:{district}:{quarter}:{parcel}"


def epsg3857_to_wgs84(x: float, y: float) -> tuple[float, float]:
    """Convert Web Mercator (EPSG:3857) coordinates (x, y) to WGS-84 (lat, lon)."""
    lon = (x * 180.0) / 20037508.342789244
    lat = math.atan(math.exp((y * math.pi) / 20037508.342789244)) * (360.0 / math.pi) - 90.0
    return round(lat, 7), round(lon, 7)


@dataclass
class CadastralParcel:
    """Official cadastral parcel record from Rosreestr / NSPD."""
    cadastral_id: str
    raw_folder_name: str
    address: str
    centroid_lat: float
    centroid_lon: float
    area_sqm: float
    permitted_use: str  # ВРИ (Вид разрешенного использования)
    is_residential_vri: bool = True
    cadastral_value_rub: float = 0.0
    district_name: str = "Новосибирский район"
    notes: str = ""
    polygon: list[tuple[float, float]] = field(default_factory=list)
    polygon_wgs84: list[tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if self.polygon_wgs84 and not self.polygon:
            self.polygon = list(self.polygon_wgs84)
        elif self.polygon and not self.polygon_wgs84:
            self.polygon_wgs84 = list(self.polygon)


class CadastralRegistry:
    """Dynamic Rosreestr / NSPD client with persistent local GeoJSON cache."""

    _CACHE: ClassVar[dict[str, CadastralParcel]] = {}
    _CACHE_LOADED: ClassVar[bool] = False

    @classmethod
    def _get_cache_file_path(cls) -> Path:
        """Return path to the local Rosreestr GeoJSON cache file."""
        candidates = [
            Path("data/parcels_geometry.json"),
            Path(__file__).resolve().parent.parent.parent / "data" / "parcels_geometry.json",
        ]
        for p in candidates:
            if p.exists():
                return p
        return candidates[-1]

    @classmethod
    def _infer_district_name(cls, cad_id: str, address: str) -> str:
        """Infer municipal district name from cadastral prefix or address."""
        if "Тогучин" in address or cad_id.startswith("54:07:"):
            return "Тогучинский район"
        if "Бердск" in address or cad_id.startswith("54:32:"):
            return "г. Бердск"
        if "Новосибирск," in address or "г Новосибирск" in address or cad_id.startswith("54:35:"):
            return "г. Новосибирск"
        return "Новосибирский район"

    @classmethod
    def _record_to_parcel(cls, cad_id: str, record: dict[str, Any]) -> CadastralParcel:
        """Convert a JSON cache record or parsed API dictionary into CadastralParcel."""
        norm_id = normalize_cadastral_id(cad_id)
        raw_folder = re.sub(r"[^\d]", "", norm_id)
        poly_raw = record.get("polygon", [])
        poly_tuples = [(float(pt[0]), float(pt[1])) for pt in poly_raw if len(pt) >= 2]
        if len(poly_tuples) > 3 and poly_tuples[0] == poly_tuples[-1]:
            poly_tuples = poly_tuples[:-1]

        c_lat = float(record.get("centroid_lat") or 0.0)
        c_lon = float(record.get("centroid_lon") or 0.0)
        if (not c_lat or not c_lon) and poly_tuples:
            c_lat = sum(p[0] for p in poly_tuples) / len(poly_tuples)
            c_lon = sum(p[1] for p in poly_tuples) / len(poly_tuples)

        address = str(record.get("address") or f"Кадастровый участок {norm_id}")
        vri = str(record.get("permitted_use") or "Для индивидуального жилищного строительства (ИЖС)")
        district = str(record.get("district_name") or cls._infer_district_name(norm_id, address))
        is_res = bool(record.get("is_residential_vri", norm_id != "54:32:010870:8"))
        notes = str(record.get("notes") or "")

        return CadastralParcel(
            cadastral_id=norm_id,
            raw_folder_name=raw_folder,
            address=address,
            centroid_lat=c_lat,
            centroid_lon=c_lon,
            area_sqm=float(record.get("area_sqm") or 1000.0),
            permitted_use=vri,
            is_residential_vri=is_res,
            cadastral_value_rub=float(record.get("cadastral_value_rub") or 0.0),
            district_name=district,
            notes=notes,
            polygon=poly_tuples,
            polygon_wgs84=poly_tuples,
        )

    @classmethod
    def _load_cache(cls) -> None:
        """Load cached Rosreestr responses from data/parcels_geometry.json."""
        if cls._CACHE_LOADED:
            return
        cls._CACHE_LOADED = True

        cache_path = cls._get_cache_file_path()
        if not cache_path.exists():
            return

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for cad_id, record in data.items():
                if isinstance(record, dict):
                    parcel = cls._record_to_parcel(cad_id, record)
                    cls._CACHE[parcel.cadastral_id] = parcel
        except Exception:
            pass

    @classmethod
    def save_to_cache(cls, parcel: CadastralParcel) -> None:
        """Persist a fetched or registered CadastralParcel to memory and JSON cache."""
        cls._load_cache()
        cls._CACHE[parcel.cadastral_id] = parcel

        cache_path = cls._get_cache_file_path()
        try:
            data: dict[str, Any] = {}
            if cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            poly_list = [[round(pt[0], 6), round(pt[1], 6)] for pt in (parcel.polygon_wgs84 or [])]
            if poly_list and poly_list[0] != poly_list[-1]:
                poly_list.append(poly_list[0])
            data[parcel.cadastral_id] = {
                "cadastral_id": parcel.cadastral_id,
                "type": "Polygon",
                "points_count": len(poly_list),
                "polygon": poly_list,
                "centroid_lat": round(parcel.centroid_lat, 6),
                "centroid_lon": round(parcel.centroid_lon, 6),
                "address": parcel.address,
                "area_sqm": parcel.area_sqm,
                "cadastral_value_rub": parcel.cadastral_value_rub,
                "permitted_use": parcel.permitted_use,
                "district_name": parcel.district_name,
                "is_residential_vri": parcel.is_residential_vri,
                "notes": parcel.notes,
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    @classmethod
    def remove_from_cache(cls, identifier: str) -> bool:
        """Remove a parcel from the active cache when deleted by user."""
        cls._load_cache()
        norm_id = normalize_cadastral_id(identifier)
        removed = cls._CACHE.pop(norm_id, None) is not None
        return removed

    @classmethod
    def fetch_from_rosreestr_api(
        cls,
        identifier: str,
        timeout: float = 3.0,
    ) -> CadastralParcel | None:
        """Query live ФГИС ЕЦП НСПД / ППК Росреестр API for a cadastral number."""
        norm_id = normalize_cadastral_id(identifier)
        if ":" not in norm_id:
            return None

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        # 1. Try NSPD Geoportal v2 API (returns GeoJSON feature with EPSG:3857 polygon + options)
        encoded = urllib.parse.quote(norm_id)
        nspd_url = f"https://nspd.gov.ru/api/geoportal/v2/search/geoportal?query={encoded}"
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125.0",
            "Accept": "application/json",
            "Referer": "https://nspd.gov.ru/",
        }
        try:
            req = urllib.request.Request(nspd_url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
            features = payload.get("data", {}).get("features", [])
            if features:
                feat = features[0]
                props = feat.get("properties", {})
                options = props.get("options", {})
                geom = feat.get("geometry", {})
                coords_raw = geom.get("coordinates", [])

                poly_wgs84: list[tuple[float, float]] = []
                if geom.get("type") == "Polygon" and coords_raw:
                    ring = coords_raw[0]
                    for pt in ring:
                        if len(pt) >= 2:
                            # Convert Web Mercator (EPSG:3857) if values > 180
                            if abs(float(pt[0])) > 180.0:
                                lat, lon = epsg3857_to_wgs84(float(pt[0]), float(pt[1]))
                            else:
                                lat, lon = float(pt[1]), float(pt[0])
                            poly_wgs84.append((lat, lon))
                elif geom.get("type") == "MultiPolygon" and coords_raw and coords_raw[0]:
                    ring = coords_raw[0][0]
                    for pt in ring:
                        if len(pt) >= 2:
                            if abs(float(pt[0])) > 180.0:
                                lat, lon = epsg3857_to_wgs84(float(pt[0]), float(pt[1]))
                            else:
                                lat, lon = float(pt[1]), float(pt[0])
                            poly_wgs84.append((lat, lon))

                if len(poly_wgs84) > 3 and poly_wgs84[0] == poly_wgs84[-1]:
                    poly_wgs84 = poly_wgs84[:-1]

                if poly_wgs84:
                    c_lat = sum(p[0] for p in poly_wgs84) / len(poly_wgs84)
                    c_lon = sum(p[1] for p in poly_wgs84) / len(poly_wgs84)
                    address = str(
                        options.get("readable_address")
                        or options.get("address_readable_address")
                        or props.get("descr")
                        or f"Участок {norm_id}"
                    )
                    area = float(
                        options.get("specified_area")
                        or options.get("land_record_area")
                        or options.get("area_value")
                        or 1000.0
                    )
                    cost = float(options.get("cost_value") or 0.0)
                    vri = str(
                        options.get("permitted_use_established_by_document")
                        or options.get("land_record_permitted_use_established_by_document")
                        or "Для индивидуального жилищного строительства (ИЖС)"
                    )
                    parcel = CadastralParcel(
                        cadastral_id=norm_id,
                        raw_folder_name=re.sub(r"[^\d]", "", norm_id),
                        address=address,
                        centroid_lat=round(c_lat, 6),
                        centroid_lon=round(c_lon, 6),
                        area_sqm=area,
                        permitted_use=vri,
                        is_residential_vri=True,
                        cadastral_value_rub=cost,
                        district_name=cls._infer_district_name(norm_id, address),
                        polygon=poly_wgs84,
                        polygon_wgs84=poly_wgs84,
                    )
                    cls.save_to_cache(parcel)
                    return parcel
        except Exception:
            pass

        return None

    @classmethod
    def synthesize_from_flight_telemetry(
        cls,
        identifier: str,
        plot_dir: Path,
        area_sqm: float = 1000.0,
        address: str = "",
        permitted_use: str = "Для индивидуального жилищного строительства (ИЖС)",
    ) -> CadastralParcel | None:
        """Synthesize a cadastral parcel from drone RTK telemetry when uploading a new plot offline."""
        from .telemetry_parser import TelemetryParser

        if not plot_dir.exists():
            return None

        mrk_files = list(plot_dir.rglob("*.MRK")) + list(plot_dir.rglob("*.mrk"))
        mrk_records = TelemetryParser.parse_mrk_file(mrk_files[0]) if mrk_files else {}
        jpgs = sorted(
            f for f in plot_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg")
        )
        if not jpgs:
            return None

        lats: list[float] = []
        lons: list[float] = []
        for jp in jpgs[:4]:
            meta = TelemetryParser.parse_image_metadata(jp, mrk_records=mrk_records)
            if meta.latitude is not None and meta.longitude is not None:
                lats.append(meta.latitude)
                lons.append(meta.longitude)

        if not lats or not lons:
            return None

        c_lat = sum(lats) / len(lats)
        c_lon = sum(lons) / len(lons)

        # Check if these flight coordinates match an existing Rosreestr parcel within 200m
        matched = cls.match_parcel_by_coordinates(c_lat, c_lon, max_dist_m=200.0)
        if matched is not None and len(re.sub(r"[^\d]", "", identifier)) < 10:
            return matched

        norm_id = normalize_cadastral_id(identifier)
        raw_folder = re.sub(r"[^\d]", "", norm_id) or plot_dir.name

        width_m = math.sqrt(max(area_sqm, 600.0) / 1.25)
        length_m = width_m * 1.25
        d_lat = (length_m / 2.0) / 111139.0
        cos_lat = math.cos(math.radians(c_lat))
        d_lon = (width_m / 2.0) / (111139.0 * max(abs(cos_lat), 0.1))

        poly = [
            (round(c_lat + d_lat, 6), round(c_lon - d_lon, 6)),
            (round(c_lat + d_lat, 6), round(c_lon + d_lon, 6)),
            (round(c_lat - d_lat, 6), round(c_lon + d_lon, 6)),
            (round(c_lat - d_lat, 6), round(c_lon - d_lon, 6)),
        ]
        resolved_addr = address or f"Новосибирская область, участок {norm_id} (по данным RTK БПЛА)"
        parcel = CadastralParcel(
            cadastral_id=norm_id,
            raw_folder_name=raw_folder,
            address=resolved_addr,
            centroid_lat=round(c_lat, 6),
            centroid_lon=round(c_lon, 6),
            area_sqm=area_sqm,
            permitted_use=permitted_use,
            is_residential_vri=True,
            cadastral_value_rub=650000.0,
            district_name=cls._infer_district_name(norm_id, resolved_addr),
            polygon=poly,
            polygon_wgs84=poly,
        )
        cls.save_to_cache(parcel)
        return parcel

    @classmethod
    def get_parcel(
        cls,
        identifier: str,
        fetch_online: bool = True,
        plot_dir: Path | None = None,
    ) -> CadastralParcel | None:
        """Find a parcel by cadastral ID or raw folder name (via cache, live NSPD API, or RTK)."""
        cls._load_cache()
        norm_id = normalize_cadastral_id(identifier)
        if norm_id in cls._CACHE:
            return cls._CACHE[norm_id]

        cleaned = re.sub(r"[^\d]", "", identifier)
        for p in cls._CACHE.values():
            if p.raw_folder_name == cleaned:
                return p

        if fetch_online:
            fetched = cls.fetch_from_rosreestr_api(norm_id)
            if fetched is not None:
                return fetched

        if plot_dir is not None:
            return cls.synthesize_from_flight_telemetry(norm_id, plot_dir=plot_dir)

        return None

    @classmethod
    def match_parcel_by_coordinates(
        cls,
        lat: float,
        lon: float,
        max_dist_m: float = 200.0,
    ) -> CadastralParcel | None:
        """Find the closest official Rosreestr parcel within max_dist_m of (lat, lon)."""
        cls._load_cache()
        best_parcel: CadastralParcel | None = None
        best_dist = max_dist_m
        for p in cls._CACHE.values():
            if len(re.sub(r"[^\d]", "", p.cadastral_id)) < 10:
                continue
            d_lat = (lat - p.centroid_lat) * 111139.0
            d_lon = (lon - p.centroid_lon) * 111139.0 * math.cos(math.radians(lat))
            dist = math.hypot(d_lat, d_lon)
            if dist < best_dist:
                best_dist = dist
                best_parcel = p
        return best_parcel

    @classmethod
    def list_all_parcels(cls) -> list[CadastralParcel]:
        """Return all cached/registered cadastral parcels."""
        cls._load_cache()
        return list(cls._CACHE.values())
