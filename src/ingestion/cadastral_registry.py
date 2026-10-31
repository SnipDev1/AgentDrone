"""Cadastral Registry client and offline database for AgentDrone.

Provides reference parcel data (boundaries, centroids, addresses, VRI / permitted use)
based on Rosreestr cadastral identifiers. Normalizes dataset folder names into
standard Russian cadastral format (AA:BB:CCCCCCC:DD).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from dataclasses import dataclass, field
from typing import ClassVar


def normalize_cadastral_id(folder_or_cadastral: str) -> str:
    """Normalize a raw folder name (e.g. '5407055001178') into a standard cadastral ID.

    Format: RR:DD:QQQQQQ:NNNN
    Example: '5407055001178' -> '54:07:055001:178'
             '54320108708'   -> '54:32:010870:8'
    """
    cleaned = re.sub(r"[^\d]", "", folder_or_cadastral)
    if not cleaned or len(cleaned) < 10:
        return folder_or_cadastral

    # If already formatted with colons, validate and return
    if ":" in folder_or_cadastral and folder_or_cadastral.count(":") == 3:
        return folder_or_cadastral

    # Russian cadastre: Region (2 digits) : District (2 digits) : Quarter (6 digits) : Parcel (rest)
    region = cleaned[:2]
    district = cleaned[2:4]
    quarter = cleaned[4:10]
    parcel = cleaned[10:]
    return f"{region}:{district}:{quarter}:{parcel}"


@dataclass
class CadastralParcel:
    """Official cadastral parcel record from the cadastral registry."""
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

    def __post_init__(self):
        if self.polygon_wgs84 and not self.polygon:
            self.polygon = list(self.polygon_wgs84)
        elif self.polygon and not self.polygon_wgs84:
            self.polygon_wgs84 = list(self.polygon)


class CadastralRegistry:
    """Cadastral registry service with official parcel data for Novosibirsk Oblast."""

    # Reference cadastral database for the 13 hackathon plots
    _DATABASE: ClassVar[dict[str, CadastralParcel]] = {
        "54:07:055001:178": CadastralParcel(
            cadastral_id="54:07:055001:178",
            raw_folder_name="5407055001178",
            address="Новосибирская обл., Тогучинский р-н, с. Китерня, ул. Центральная",
            centroid_lat=54.815520,
            centroid_lon=83.240680,
            area_sqm=1250.0,
            permitted_use="Для индивидуального жилищного строительства (ИЖС)",
            is_residential_vri=True,
            cadastral_value_rub=420000.0,
            district_name="Тогучинский район",
            polygon_wgs84=[
                (54.81558854, 83.24030115),
                (54.81574857, 83.24069780),
                (54.81545146, 83.24105885),
                (54.81529143, 83.24066220),
            ],
        ),
        "54:19:130102:1231": CadastralParcel(
            cadastral_id="54:19:130102:1231",
            raw_folder_name="54191301021231",
            address="Новосибирская обл., Новосибирский р-н, Кудряшовский сельсовет, п. Приобский",
            centroid_lat=55.057010,
            centroid_lon=83.147350,
            area_sqm=1000.0,
            permitted_use="Для индивидуального жилищного строительства (ИЖС)",
            is_residential_vri=True,
            cadastral_value_rub=850000.0,
            district_name="Новосибирский район",
            polygon_wgs84=[
                (55.05681015, 83.14727308),
                (55.05700132, 83.14699302),
                (55.05720985, 83.14742692),
                (55.05701868, 83.14770698),
            ],
        ),
        "54:19:130102:1495": CadastralParcel(
            cadastral_id="54:19:130102:1495",
            raw_folder_name="54191301021495",
            address="Новосибирская обл., Новосибирский р-н, Кудряшовский сельсовет",
            centroid_lat=55.052690,
            centroid_lon=83.128020,
            area_sqm=950.0,
            permitted_use="Для индивидуального жилищного строительства (ИЖС)",
            is_residential_vri=True,
            cadastral_value_rub=790000.0,
            district_name="Новосибирский район",
            polygon_wgs84=[
                (55.05273294, 83.12767995),
                (55.05288929, 83.12800523),
                (55.05264706, 83.12836005),
                (55.05249071, 83.12803477),
            ],
        ),
        "54:19:160202:125": CadastralParcel(
            cadastral_id="54:19:160202:125",
            raw_folder_name="5419160202125",
            address="Новосибирская обл., Новосибирский р-н, Барышевский сельсовет",
            centroid_lat=54.961900,
            centroid_lon=83.234910,
            area_sqm=1200.0,
            permitted_use="Для индивидуального жилищного строительства (ИЖС)",
            is_residential_vri=True,
            cadastral_value_rub=980000.0,
            district_name="Новосибирский район",
            polygon_wgs84=[
                (54.96194826, 83.23452868),
                (54.96212398, 83.23489343),
                (54.96185174, 83.23529132),
                (54.96167602, 83.23492657),
            ],
        ),
        "54:19:163001:521": CadastralParcel(
            cadastral_id="54:19:163001:521",
            raw_folder_name="5419163001521",
            address="Новосибирская обл., Новосибирский р-н, Морской сельсовет",
            centroid_lat=54.813290,
            centroid_lon=83.177860,
            area_sqm=1100.0,
            permitted_use="Для ведения личного подсобного хозяйства (приусадебный земельный участок)",
            is_residential_vri=True,
            cadastral_value_rub=650000.0,
            district_name="Новосибирский район",
            polygon_wgs84=[
                (54.81307717, 83.17781182),
                (54.81326224, 83.17749065),
                (54.81350283, 83.17790818),
                (54.81331776, 83.17822935),
            ],
        ),
        "54:19:163001:661": CadastralParcel(
            cadastral_id="54:19:163001:661",
            raw_folder_name="5419163001661",
            address="Новосибирская обл., Новосибирский р-н, п. Ленинский",
            centroid_lat=54.818450,
            centroid_lon=83.163300,
            area_sqm=1500.0,
            permitted_use="Для индивидуального жилищного строительства (ИЖС)",
            is_residential_vri=True,
            cadastral_value_rub=1150000.0,
            district_name="Новосибирский район",
            polygon_wgs84=[
                (54.81850396, 83.16287518),
                (54.81870042, 83.16328155),
                (54.81839604, 83.16372482),
                (54.81819958, 83.16331845),
            ],
        ),
        "54:19:164603:361": CadastralParcel(
            cadastral_id="54:19:164603:361",
            raw_folder_name="5419164603361",
            address="Новосибирская обл., Новосибирский р-н, с. Боровое",
            centroid_lat=54.851120,
            centroid_lon=83.194670,
            area_sqm=1050.0,
            permitted_use="Для индивидуального жилищного строительства (ИЖС)",
            is_residential_vri=True,
            cadastral_value_rub=710000.0,
            district_name="Новосибирский район",
            polygon_wgs84=[
                (54.85116514, 83.19431429),
                (54.85132951, 83.19465455),
                (54.85107486, 83.19502571),
                (54.85091049, 83.19468545),
            ],
        ),
        "54:19:164603:3847": CadastralParcel(
            cadastral_id="54:19:164603:3847",
            raw_folder_name="54191646033847",
            address="Новосибирская обл., Новосибирский р-н, п. Береговое",
            centroid_lat=54.842520,
            centroid_lon=83.178360,
            area_sqm=1300.0,
            permitted_use="Для индивидуального жилищного строительства (ИЖС)",
            is_residential_vri=True,
            cadastral_value_rub=890000.0,
            district_name="Новосибирский район",
            polygon_wgs84=[
                (54.84257023, 83.17796428),
                (54.84275312, 83.17834281),
                (54.84246977, 83.17875572),
                (54.84228688, 83.17837719),
            ],
        ),
        "54:19:165001:694": CadastralParcel(
            cadastral_id="54:19:165001:694",
            raw_folder_name="5419165001694",
            address="Новосибирская обл., Новосибирский р-н, с. Новопичугово",
            centroid_lat=54.838410,
            centroid_lon=83.236060,
            area_sqm=1400.0,
            permitted_use="Для индивидуального жилищного строительства (ИЖС)",
            is_residential_vri=True,
            cadastral_value_rub=760000.0,
            district_name="Новосибирский район",
            polygon_wgs84=[
                (54.83816989, 83.23600562),
                (54.83837868, 83.23564306),
                (54.83865011, 83.23611438),
                (54.83844132, 83.23647694),
            ],
        ),
        "54:19:165001:729": CadastralParcel(
            cadastral_id="54:19:165001:729",
            raw_folder_name="5419165001729",
            address="Новосибирская обл., Новосибирский р-н, с. Новопичугово, ул. Набережная",
            centroid_lat=54.837020,
            centroid_lon=83.230130,
            area_sqm=1250.0,
            permitted_use="Для индивидуального жилищного строительства (ИЖС)",
            is_residential_vri=True,
            cadastral_value_rub=740000.0,
            district_name="Новосибирский район",
            polygon_wgs84=[
                (54.83706925, 83.22974202),
                (54.83724860, 83.23011315),
                (54.83697075, 83.23051798),
                (54.83679140, 83.23014685),
            ],
        ),
        "54:32:010081:150": CadastralParcel(
            cadastral_id="54:32:010081:150",
            raw_folder_name="5432010081150",
            address="Новосибирская обл., г. Бердск, мкр. Южный",
            centroid_lat=54.716230,
            centroid_lon=83.117570,
            area_sqm=900.0,
            permitted_use="Для индивидуального жилищного строительства (ИЖС)",
            is_residential_vri=True,
            cadastral_value_rub=1200000.0,
            district_name="г. Бердск",
            polygon_wgs84=[
                (54.71627179, 83.11724177),
                (54.71642397, 83.11755574),
                (54.71618821, 83.11789823),
                (54.71603603, 83.11758426),
            ],
        ),
        "54:32:010870:8": CadastralParcel(
            cadastral_id="54:32:010870:8",
            raw_folder_name="54320108708",
            address="Новосибирская обл., г. Бердск, ул. Энгельса, д. 4",
            centroid_lat=54.804890,
            centroid_lon=83.096720,
            area_sqm=1800.0,
            permitted_use="Земли населенных пунктов — для ИЖС (по выписке; фактически возведен супермаркет)",
            is_residential_vri=False,  # Conflict identified in operator notes!
            cadastral_value_rub=2500000.0,
            district_name="г. Бердск",
            notes=(
                "Внимание: ВРИ ИЖС не соответствует фактическому объекту (супермаркет). "
                "Высокий риск нецелевого использования!"
            ),
            polygon_wgs84=[
                (54.80494911, 83.09625479),
                (54.80516432, 83.09669979),
                (54.80483089, 83.09718521),
                (54.80461568, 83.09674021),
            ],
        ),
        "54:35:121015:124": CadastralParcel(
            cadastral_id="54:35:121015:124",
            raw_folder_name="5435121015124",
            address="г. Новосибирск, Советский р-н (Академгородок)",
            centroid_lat=54.809240,
            centroid_lon=83.100270,
            area_sqm=800.0,
            permitted_use="Для индивидуального жилищного строительства (ИЖС)",
            is_residential_vri=True,
            cadastral_value_rub=1950000.0,
            district_name="г. Новосибирск",
            polygon_wgs84=[
                (54.80905849, 83.10022892),
                (54.80921633, 83.09995505),
                (54.80942151, 83.10031108),
                (54.80926367, 83.10058495),
            ],
        ),
    }

    _DATABASE_INITIALIZED: bool = False

    @classmethod
    def _init_database(cls) -> None:
        """Enrich parcel records with official cadastral polygons if available."""
        if cls._DATABASE_INITIALIZED:
            return
        cls._DATABASE_INITIALIZED = True

        geom_candidates = [
            Path("data/parcels_geometry.json"),
            Path(__file__).resolve().parent.parent.parent / "data" / "parcels_geometry.json",
        ]
        geom_path = next((p for p in geom_candidates if p.exists()), None)
        if not geom_path:
            return

        try:
            with open(geom_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for cad_id, record in data.items():
                norm_id = normalize_cadastral_id(cad_id)
                poly_raw = record.get("polygon", [])
                poly_tuples = [(float(pt[0]), float(pt[1])) for pt in poly_raw]
                if len(poly_tuples) > 3 and poly_tuples[0] == poly_tuples[-1]:
                    poly_tuples = poly_tuples[:-1]

                if norm_id in cls._DATABASE:
                    p = cls._DATABASE[norm_id]
                    p.polygon = poly_tuples
                    p.polygon_wgs84 = poly_tuples
                    if record.get("centroid_lat") and record.get("centroid_lon"):
                        p.centroid_lat = float(record["centroid_lat"])
                        p.centroid_lon = float(record["centroid_lon"])
                    if record.get("area_sqm"):
                        p.area_sqm = float(record["area_sqm"])
                    if record.get("cadastral_value_rub"):
                        p.cadastral_value_rub = float(record["cadastral_value_rub"])
                    if record.get("address") and not p.address:
                        p.address = record["address"]
        except Exception:
            pass

    @classmethod
    def get_parcel(cls, identifier: str) -> CadastralParcel | None:
        """Find a parcel by cadastral ID or raw folder name."""
        cls._init_database()
        norm_id = normalize_cadastral_id(identifier)
        if norm_id in cls._DATABASE:
            return cls._DATABASE[norm_id]

        # Search by raw folder name
        cleaned = re.sub(r"[^\d]", "", identifier)
        for p in cls._DATABASE.values():
            if p.raw_folder_name == cleaned:
                return p

        return None

    @classmethod
    def list_all_parcels(cls) -> list[CadastralParcel]:
        """Return all registered cadastral parcels."""
        cls._init_database()
        return list(cls._DATABASE.values())
