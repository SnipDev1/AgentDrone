"""Bank Credit Registry repository for AgentDrone.

Loads the exact customer CSV/XLSX table (`data/registry.csv`) without inventing
any extra fields:
- Территория (`address`)
- Кадастровый номер (`cadastral_id`)
- Плановая дата (`planned_completion_date`)
- Фактическая дата (`actual_completion_date`)
- Наличие ПЗ (`has_pz`)
- ОН зарегистрирован (`is_registered`)
- Залог (`is_pledged`)
- Дата начала строительства (`construction_start_date`)
- Наличие фото с неба (`has_aerial_photo`)
"""

from __future__ import annotations

import csv
import io
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

from ..models.schemas import CreditObject
from .cadastral_registry import CadastralRegistry, normalize_cadastral_id

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_REGISTRY_PATH = PROJECT_ROOT / "data" / "registry.csv"


def _digits_only(val: str) -> str:
    return re.sub(r"\D", "", val or "")


def _load_csv_registry(
    csv_path: Path = CSV_REGISTRY_PATH,
) -> tuple[list[CreditObject], dict[str, CreditObject], dict[str, CreditObject]]:
    rows_list: list[CreditObject] = []
    by_norm: dict[str, CreditObject] = {}
    by_digits: dict[str, CreditObject] = {}
    if not csv_path.exists():
        return rows_list, by_norm, by_digits
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_cad = (row.get("Кадастровый номер") or "").strip()
                if not raw_cad:
                    continue
                norm_id = raw_cad if ":" in raw_cad else normalize_cadastral_id(raw_cad)
                digits_id = _digits_only(raw_cad)
                has_pz_val = (row.get("Наличие ПЗ") or "Нет").strip()
                obj = CreditObject(
                    cadastral_id=norm_id,
                    address=(row.get("Территория") or "").strip(),
                    planned_completion_date=(row.get("Плановая дата") or "Нет").strip(),
                    actual_completion_date=(row.get("Фактическая дата") or "Нет").strip(),
                    has_pz=has_pz_val,
                    is_registered=(row.get("ОН зарегистрирован") or "Нет").strip(),
                    is_pledged=(row.get("Залог") or "Нет").strip(),
                    construction_start_date=(row.get("Дата начала строительства") or "Нет").strip(),
                    has_aerial_photo=(row.get("Наличие фото с неба") or "Да").strip(),
                    days_overdue=30 if has_pz_val.lower() == "да" else 0,
                    is_minimal_schema=True,
                )
                rows_list.append(obj)
                if norm_id not in by_norm:
                    by_norm[norm_id] = obj
                if digits_id and digits_id not in by_digits:
                    by_digits[digits_id] = obj
    except Exception:
        pass
    return rows_list, by_norm, by_digits


def _xlsx_bytes_to_csv_text(xlsx_bytes: bytes) -> str:
    """Convert standard single-sheet .xlsx bytes to CSV text using stdlib zipfile + xml."""
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as z:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            ss_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in ss_root.findall("main:si", ns):
                texts = [t.text or "" for t in si.findall(".//main:t", ns)]
                shared_strings.append("".join(texts))

        sheet_root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows_out: list[list[str]] = []
        for row_el in sheet_root.findall(".//main:sheetData/main:row", ns):
            row_vals: list[str] = []
            for c_el in row_el.findall("main:c", ns):
                c_type = c_el.attrib.get("t", "")
                v_el = c_el.find("main:v", ns)
                val_raw = v_el.text if v_el is not None and v_el.text is not None else ""
                if c_type == "s" and val_raw.isdigit():
                    idx = int(val_raw)
                    val_str = shared_strings[idx] if 0 <= idx < len(shared_strings) else val_raw
                elif c_type == "" and val_raw.replace(".", "", 1).isdigit():
                    # Check if Excel serial date (between 30000 and 60000)
                    num = float(val_raw)
                    if 30000 <= num <= 60000 and num.is_integer():
                        dt = datetime(1899, 12, 30) + timedelta(days=int(num))
                        val_str = dt.strftime("%d.%m.%Y")
                    else:
                        val_str = val_raw
                else:
                    val_str = val_raw
                row_vals.append(val_str.strip())
            if any(row_vals):
                rows_out.append(row_vals)

    out_io = io.StringIO()
    writer = csv.writer(out_io)
    for r in rows_out:
        writer.writerow(r)
    return out_io.getvalue()


CREDIT_REGISTRY_LIST, CREDIT_REGISTRY_DATA, CREDIT_REGISTRY_DIGITS = _load_csv_registry()


class CreditRegistry:
    """Service to access customer registry records for monitored collateral plots."""

    @classmethod
    def reload(cls) -> None:
        """Reload registry from data/registry.csv."""
        global CREDIT_REGISTRY_LIST, CREDIT_REGISTRY_DATA, CREDIT_REGISTRY_DIGITS
        CREDIT_REGISTRY_LIST, CREDIT_REGISTRY_DATA, CREDIT_REGISTRY_DIGITS = _load_csv_registry()

    @classmethod
    def update_from_file_bytes(cls, file_bytes: bytes, filename: str = "registry.csv") -> int:
        """Save uploaded CSV or XLSX table to data/registry.csv and reload."""
        if filename.lower().endswith(".xlsx"):
            text = _xlsx_bytes_to_csv_text(file_bytes)
        else:
            text = file_bytes.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames or "Кадастровый номер" not in reader.fieldnames:
            raise ValueError("Файл реестра должен содержать колонку 'Кадастровый номер'")
        CSV_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        CSV_REGISTRY_PATH.write_text(text, encoding="utf-8")
        cls.reload()
        return len(CREDIT_REGISTRY_LIST)

    @classmethod
    def update_from_csv_bytes(cls, csv_bytes: bytes) -> int:
        """Save uploaded CSV table to data/registry.csv and reload."""
        return cls.update_from_file_bytes(csv_bytes, "registry.csv")

    @classmethod
    def get_credit_object(cls, identifier: str) -> CreditObject:
        """Find registry record by cadastral ID or raw numeric folder name."""
        if not CREDIT_REGISTRY_DATA:
            cls.reload()
        raw_clean = (identifier or "").strip()
        if raw_clean in CREDIT_REGISTRY_DATA:
            return CREDIT_REGISTRY_DATA[raw_clean]
        norm_id = normalize_cadastral_id(raw_clean)
        if norm_id in CREDIT_REGISTRY_DATA:
            return CREDIT_REGISTRY_DATA[norm_id]
        digits = _digits_only(raw_clean)
        if digits and digits in CREDIT_REGISTRY_DIGITS:
            return CREDIT_REGISTRY_DIGITS[digits]

        # Check prefix before underscore or dash (e.g. '5419160202125_2027' -> '5419160202125')
        prefix = re.split(r"[_\- ]", raw_clean)[0]
        if prefix != raw_clean:
            pref_norm = normalize_cadastral_id(prefix)
            if pref_norm in CREDIT_REGISTRY_DATA:
                return CREDIT_REGISTRY_DATA[pref_norm]
            pref_digits = _digits_only(prefix)
            if pref_digits and pref_digits in CREDIT_REGISTRY_DIGITS:
                return CREDIT_REGISTRY_DIGITS[pref_digits]

        parcel = CadastralRegistry.get_parcel(norm_id, fetch_online=False)
        addr = parcel.address if parcel and parcel.address else "Н/Д"
        return CreditObject(
            cadastral_id=norm_id,
            address=addr,
            planned_completion_date="Нет",
            actual_completion_date="Нет",
            has_pz="Нет",
            is_registered="Нет",
            is_pledged="Нет",
            construction_start_date="Нет",
            has_aerial_photo="Да",
            is_minimal_schema=True,
        )

    @classmethod
    def list_all(cls) -> list[CreditObject]:
        """Return all registered records."""
        if not CREDIT_REGISTRY_LIST:
            cls.reload()
        return list(CREDIT_REGISTRY_LIST)
