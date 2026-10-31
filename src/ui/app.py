"""Web workspace for Sberbank Collateral Inspector (АС «Калита» / Мониторинг ИЖС).

Provides a clean, ergonomic corporate banking interface for:
- Uploading and deleting user drone inspection folders (ZIP or JPG/MRK files)
- Dynamic Rosreestr / NSPD cadastral data lookup (boundaries, area, VRI, value)
- Photogrammetric red cadastral boundary overlay across all aerial shots (Shots 1..N)
- Automated construction readiness assessment (0% / 50% / 100%) and strategy selection
- Generating official Collateral Inspection Acts (Appendix 1, HTML/PDF)
"""

from __future__ import annotations

import base64
import importlib
import io
from pathlib import Path
import sys

from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as _mod_cfg  # noqa: E402
import src.cv.boundary_overlay as _mod_bo  # noqa: E402
import src.cv.feature_extractor as _mod_fe  # noqa: E402
import src.cv.fraud_detector as _mod_fd  # noqa: E402
import src.cv.material_classifier as _mod_mc  # noqa: E402
import src.cv.multispectral as _mod_ms  # noqa: E402
import src.cv.pipeline as _mod_cp  # noqa: E402
import src.cv.readiness_classifier as _mod_rc  # noqa: E402
import src.dataset as _mod_ds  # noqa: E402
import src.dataset_manager as _mod_dm  # noqa: E402
import src.engine.decision_engine as _mod_de  # noqa: E402
import src.gigachat_service as _mod_gp  # noqa: E402
import src.gigachat_service.prompts as _mod_gprompts  # noqa: E402
import src.gigachat_service.service as _mod_gs  # noqa: E402
import src.ingestion as _mod_ing  # noqa: E402
import src.ingestion.angle_detector as _mod_ad  # noqa: E402
import src.ingestion.cadastral_registry as _mod_cr  # noqa: E402
import src.ingestion.cadastral_validator as _mod_cv  # noqa: E402
import src.ingestion.credit_registry as _mod_cred_reg  # noqa: E402
import src.ingestion.notes_parser as _mod_np  # noqa: E402
import src.ingestion.ppk_parser as _mod_ppk  # noqa: E402
import src.ingestion.telemetry_parser as _mod_tp  # noqa: E402
import src.models as _mod_models  # noqa: E402
import src.models.credit as _mod_mcred  # noqa: E402
import src.models.inspection as _mod_minsp  # noqa: E402
import src.models.schemas as _mod_mschemas  # noqa: E402
import src.models.strategy as _mod_mstrat  # noqa: E402
import src.report as _mod_rp  # noqa: E402
import src.report.builder as _mod_rb  # noqa: E402
import src.report.generator as _mod_rg  # noqa: E402
import src.ui.sber_theme as _mod_st  # noqa: E402

if "pytest" not in sys.modules and getattr(sys, "_agentdrone_reloaded_v18", False) is False:
    for _m_schema in (_mod_mschemas, _mod_mcred, _mod_minsp, _mod_mstrat, _mod_models):
        try:
            importlib.reload(_m_schema)
        except Exception:
            pass
    for _m in (
        _mod_cfg,
        _mod_st,
        _mod_cr,
        _mod_cred_reg,
        _mod_tp,
        _mod_ad,
        _mod_cv,
        _mod_np,
        _mod_ppk,
        _mod_ing,
        _mod_de,
        _mod_gprompts,
        _mod_gs,
        _mod_gp,
        _mod_fe,
        _mod_rc,
        _mod_mc,
        _mod_fd,
        _mod_ms,
        _mod_cp,
        _mod_bo,
        _mod_dm,
        _mod_ds,
        _mod_rb,
        _mod_rg,
        _mod_rp,
    ):
        try:
            importlib.reload(_m)
        except Exception:
            pass
    try:
        st.cache_data.clear()
    except Exception:
        pass
    sys._agentdrone_reloaded_v18 = True  # type: ignore[attr-defined]

from src.cv.boundary_overlay import BoundaryOverlay  # noqa: E402
from src.cv.pipeline import CVPipeline  # noqa: E402
from src.dataset import DroneDataset  # noqa: E402
from src.dataset_manager import ParcelManager  # noqa: E402
from src.engine.decision_engine import DecisionEngine  # noqa: E402
from src.gigachat_service.service import GigaChatInspectorService  # noqa: E402
from src.ingestion.cadastral_registry import (  # noqa: E402
    CadastralParcel,
    CadastralRegistry,
    normalize_cadastral_id,
)
from src.ingestion.credit_registry import CreditRegistry  # noqa: E402
from src.models.schemas import (  # noqa: E402
    CreditObject,
    CVInspectionResult,
    DroneImageMeta,
    StrategyRecommendation,
)
from src.models.strategy import ReportPhotoItem  # noqa: E402
from src.report.builder import build_act_from_cv  # noqa: E402
from src.report.generator import ActGenerator  # noqa: E402
from src.ui.sber_theme import (  # noqa: E402
    get_sber_css,
    render_kpi_card,
    render_sber_header,
)

st.set_page_config(
    page_title="АС Калита — Мониторинг залогов ИЖС",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _render_html(raw_html: str) -> None:
    """Render HTML via st.markdown after stripping line indentation to prevent code-block parsing."""
    compact = "\n".join(line.strip() for line in raw_html.splitlines() if line.strip())
    st.markdown(compact, unsafe_allow_html=True)


def apply_custom_css() -> None:
    """Apply corporate banking stylesheet and clean layout rules."""
    _render_html(get_sber_css())
    _render_html(
        """
        <style>
        .ws-panel {
            background: #FFFFFF;
            border: 1px solid #DDE3EA;
            border-radius: 10px;
            padding: 16px 18px;
            margin-bottom: 14px;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
        }
        .ws-panel-header {
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #475569;
            border-bottom: 1px solid #E2E8F0;
            padding-bottom: 8px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .ws-kv-grid {
            display: grid;
            grid-template-columns: 165px 1fr;
            row-gap: 6px;
            column-gap: 12px;
            font-size: 0.85rem;
        }
        .ws-kv-label {
            color: #64748B;
            font-weight: 500;
        }
        .ws-kv-val {
            color: #0F172A;
            font-weight: 600;
        }
        .ws-status-pill {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
        }
        .ws-status-ok { background: #DCFCE7; color: #166534; }
        .ws-status-warn { background: #FEF3C7; color: #92400E; }
        .ws-status-err { background: #FEE2E2; color: #991B1B; }
        </style>
        """
    )


def get_available_plot_ids() -> list[str]:
    """Retrieve list of available cadastral plot IDs on the server."""
    return ParcelManager.list_plots()


def clear_all_caches() -> None:
    """Clear Streamlit caches and session state after upload/delete operations."""
    st.cache_data.clear()
    if "inspected_plots" in st.session_state:
        st.session_state["inspected_plots"].clear()


def handle_delete_plot(plot_id: str) -> None:
    """Delete a parcel folder and synchronize UI state."""
    try:
        deleted = ParcelManager.delete_plot(plot_id)
        if deleted:
            CadastralRegistry.remove_from_cache(plot_id)
            clear_all_caches()
            st.session_state["pending_plot_id"] = None
            st.toast(f"Объект {normalize_cadastral_id(plot_id)} удален с сервера.")
            st.rerun()
        else:
            st.error(f"Не удалось найти папку участка {plot_id} для удаления.")
    except Exception as exc:
        st.error(f"Ошибка при удалении участка {plot_id}: {exc}")


def handle_delete_all_plots() -> None:
    """Delete all parcel folders from the server."""
    try:
        count = ParcelManager.delete_all_plots()
        clear_all_caches()
        st.session_state["pending_plot_id"] = None
        st.toast(f"Удалено объектов с сервера: {count}")
        st.rerun()
    except Exception as exc:
        st.error(f"Ошибка при очистке объектов: {exc}")


def handle_upload_zip(uploaded_zip, target_plot_id: str | None = None) -> None:
    """Extract a ZIP archive of one or more parcel folders into the dataset root."""
    if uploaded_zip is None:
        st.warning("Выберите ZIP-архив для загрузки.")
        return
    try:
        zip_bytes = uploaded_zip.getvalue()
        effective_target = (
            target_plot_id.strip()
            if (target_plot_id and target_plot_id.strip())
            else getattr(uploaded_zip, "name", None)
        )
        extracted_ids = ParcelManager.upload_zip(
            zip_bytes,
            target_plot_id=effective_target,
        )
        clear_all_caches()
        if extracted_ids:
            st.session_state["pending_plot_id"] = extracted_ids[0]
            st.toast(f"Загружено участков: {len(extracted_ids)} ({', '.join(extracted_ids)})")
            st.rerun()
        else:
            st.warning("Архив не содержит файлов съемки.")
    except Exception as exc:
        st.error(f"Ошибка распаковки архива: {exc}")


def handle_upload_files(target_plot_id: str, uploaded_files: list) -> None:
    """Upload individual drone photos / MRK logs into a new or existing parcel folder."""
    if not target_plot_id or not target_plot_id.strip():
        st.error("Пожалуйста, укажите кадастровый номер для создаваемой папки.")
        return
    if not uploaded_files:
        st.error("Пожалуйста, выберите файлы съемки для загрузки.")
        return
    try:
        file_tuples = [(f.name, f.getvalue()) for f in uploaded_files]
        saved_id = ParcelManager.upload_files(target_plot_id.strip(), file_tuples)
        clear_all_caches()
        st.session_state["pending_plot_id"] = saved_id
        st.toast(f"Объект {normalize_cadastral_id(saved_id)} ({len(file_tuples)} файлов) загружен.")
        st.rerun()
    except Exception as exc:
        st.error(f"Ошибка сохранения файлов: {exc}")


@st.cache_data(show_spinner=False)
def analyze_single_plot(
    plot_id: str,
) -> tuple[CVInspectionResult, CreditObject | None, StrategyRecommendation]:
    """Run CV pipeline, Sber GigaChat-2-Max Vision AI Agent, and decision engine for a single plot (cached)."""
    ds = DroneDataset()
    plot_dir = ds.root_dir / plot_id
    parcel = CadastralRegistry.get_parcel(plot_id, fetch_online=True, plot_dir=plot_dir)
    credit_obj = CreditRegistry.get_credit_object(plot_id)
    cv_result = CVPipeline.analyze_plot(plot_id, plot_dir=plot_dir, credit_object=credit_obj)
    gigachat_srv = GigaChatInspectorService()
    if gigachat_srv.is_online:
        gigachat_srv.analyze_plot_with_vision_agent(
            cv_result=cv_result,
            credit_obj=credit_obj,
            parcel=parcel,
            plot_dir=plot_dir,
        )
    decision_result = DecisionEngine.evaluate(cv_result, credit_object=credit_obj)
    return cv_result, credit_obj, decision_result


def build_plot_summary(
    pid: str,
    inspected: tuple[CVInspectionResult, CreditObject | None, StrategyRecommendation] | None = None,
) -> dict:
    """Assemble a summary row for a plot strictly from customer CSV fields and CV results."""
    if inspected is not None:
        cv_res, cred, dec = inspected
        return {
            "raw_id": pid,
            "cadastral_id": cv_res.cadastral_id,
            "address": cred.address if cred else "Н/Д",
            "planned_date": getattr(cred, "planned_completion_date", "Нет") if cred else "Нет",
            "actual_date": getattr(cred, "actual_completion_date", "Нет") if cred else "Нет",
            "has_pz": getattr(cred, "has_pz", "Нет") if cred else "Нет",
            "is_registered": getattr(cred, "is_registered", "Нет") if cred else "Нет",
            "is_pledged": getattr(cred, "is_pledged", "Нет") if cred else "Нет",
            "start_date": getattr(cred, "construction_start_date", "Нет") if cred else "Нет",
            "has_aerial_photo": getattr(cred, "has_aerial_photo", "Да") if cred else "Да",
            "readiness_pct": cv_res.readiness_pct,
            "stage": cv_res.readiness_stage,
            "material": cv_res.detected_material.value,
            "strategy": dec.strategy.value,
            "strategy_code": dec.strategy_code,
            "risk_level": dec.risk_level,
            "borrower": getattr(cred, "borrower_name", "") if cred else "",
            "credit_amount": getattr(cred, "credit_amount_rub", 0.0) if cred else 0.0,
            "fraud_flag": cv_res.fraud_risk_flag,
            "days_overdue": getattr(cred, "days_overdue", 0) if cred else 0,
            "kalita_status": dec.kalita_status,
            "is_inspected": True,
        }

    cred = CreditRegistry.get_credit_object(pid)
    parcel = CadastralRegistry.get_parcel(pid, fetch_online=False)
    return {
        "raw_id": pid,
        "cadastral_id": (
            cred.cadastral_id
            if cred
            else (parcel.cadastral_id if parcel else normalize_cadastral_id(pid))
        ),
        "address": cred.address if cred else (parcel.address if parcel else "Н/Д"),
        "planned_date": cred.planned_completion_date if cred else "Нет",
        "actual_date": cred.actual_completion_date if cred else "Нет",
        "has_pz": cred.has_pz if cred else "Нет",
        "is_registered": cred.is_registered if cred else "Нет",
        "is_pledged": cred.is_pledged if cred else "Нет",
        "start_date": cred.construction_start_date if cred else "Нет",
        "has_aerial_photo": cred.has_aerial_photo if cred else "Да",
        "readiness_pct": None,
        "stage": "—",
        "material": "Не проверено",
        "strategy": "Ожидает выбора участка",
        "strategy_code": "ОЖИДАЕТ",
        "risk_level": "PENDING",
        "borrower": getattr(cred, "borrower_name", "") if cred else "",
        "credit_amount": getattr(cred, "credit_amount_rub", 0.0) if cred else 0.0,
        "fraud_flag": False,
        "days_overdue": getattr(cred, "days_overdue", 0) if cred else 0,
        "kalita_status": "Ожидает инспекции",
        "is_inspected": False,
    }


def get_portfolio_summary(
    plot_ids: list[str],
    inspected_map: dict[str, tuple[CVInspectionResult, CreditObject | None, StrategyRecommendation]],
) -> list[dict]:
    """Assemble portfolio summary without running CV on unvisited plots."""
    return [build_plot_summary(pid, inspected=inspected_map.get(pid)) for pid in plot_ids]


@st.cache_data(show_spinner=False)
def analyze_portfolio_summary(plot_ids: tuple[str, ...]) -> list[dict]:
    """Run batch inspection across all plots when explicitly requested by user."""
    summary = []
    for pid in plot_ids:
        try:
            inspected = analyze_single_plot(pid)
            summary.append(build_plot_summary(pid, inspected=inspected))
        except Exception:
            continue
    return summary


@st.cache_data(show_spinner=False)
def render_cached_boundary_overlay(
    img_path_str: str,
    meta_dict: dict | None,
    parcel_id: str,
    line_width: int,
    show_setback: bool = True,
) -> bytes:
    """Render and cache JPEG bytes with red cadastral boundary and 3m setback line."""
    img_path = Path(img_path_str)
    meta = DroneImageMeta(**meta_dict) if meta_dict else None
    parcel = CadastralRegistry.get_parcel(parcel_id, fetch_online=False, plot_dir=img_path.parent)
    if parcel is None:
        parcel = CadastralParcel(
            cadastral_id=normalize_cadastral_id(parcel_id),
            raw_folder_name=parcel_id,
            address="Участок ИЖС",
            centroid_lat=meta.latitude if (meta and meta.latitude) else 55.0,
            centroid_lon=meta.longitude if (meta and meta.longitude) else 83.0,
            area_sqm=1000.0,
            permitted_use="Для ИЖС",
        )

    with Image.open(img_path) as raw_img:
        overlaid = BoundaryOverlay.apply_boundary_overlay(
            image=raw_img,
            meta=meta,
            parcel=parcel,
            line_width=line_width,
            show_setback=show_setback,
        )
        overlaid.thumbnail((1400, 1050))
        buf = io.BytesIO()
        overlaid.save(buf, format="JPEG", quality=88)
        return buf.getvalue()


def get_base64_image(
    img_path: Path,
    apply_overlay: bool = False,
    meta: DroneImageMeta | None = None,
    parcel: CadastralParcel | None = None,
) -> str | None:
    """Convert image to base64 data URI for embedding into HTML/PDF Act."""
    try:
        with Image.open(img_path) as img:
            if apply_overlay and parcel is not None:
                img = BoundaryOverlay.apply_boundary_overlay(
                    image=img,
                    meta=meta,
                    parcel=parcel,
                    line_width=6,
                )
            img.thumbnail((860, 645))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=84)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def match_shots_to_metadata(
    rgb_images: list[Path],
    shots_metadata: list[DroneImageMeta],
) -> list[tuple[Path, DroneImageMeta | None]]:
    """Pair each drone photo in the folder with its exact DroneImageMeta by filename."""
    meta_by_name = {m.filename: m for m in shots_metadata}
    dji_images = [p for p in rgb_images if "DJI" in p.name.upper()]
    target_images = dji_images if dji_images else rgb_images

    paired: list[tuple[Path, DroneImageMeta | None]] = []
    for idx, img_path in enumerate(target_images):
        meta = meta_by_name.get(img_path.name)
        if meta is None and idx < len(shots_metadata):
            meta = shots_metadata[idx]
        paired.append((img_path, meta))
    return paired


def build_inspection_folium_map(
    parcel: CadastralParcel | None,
    shots_meta: list[DroneImageMeta],
    is_geofence_valid: bool = True,
) -> str:
    """Build interactive Leaflet map HTML with Rosreestr polygon and drone RTK markers."""
    import json

    center_lat = parcel.centroid_lat if parcel else 55.028
    center_lon = parcel.centroid_lon if parcel else 82.921
    if not parcel and shots_meta:
        valid_coords = [
            (m.latitude, m.longitude)
            for m in shots_meta
            if m.latitude is not None and m.longitude is not None
        ]
        if valid_coords:
            center_lat = sum(c[0] for c in valid_coords) / len(valid_coords)
            center_lon = sum(c[1] for c in valid_coords) / len(valid_coords)

    poly_coords = parcel.polygon_wgs84 if (parcel and parcel.polygon_wgs84) else []
    markers = []
    for idx, m in enumerate(shots_meta, start=1):
        if m.latitude is not None and m.longitude is not None:
            markers.append(
                {
                    "lat": m.latitude,
                    "lon": m.longitude,
                    "label": f"Кадр {idx}: {m.filename} ({m.shot_angle.value})",
                }
            )

    poly_json = json.dumps(poly_coords)
    markers_json = json.dumps(markers, ensure_ascii=False)
    cad_label = parcel.cadastral_id if parcel else "Контур ЗУ"
    stroke_color = "#DC2626" if is_geofence_valid else "#B91C1C"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body, #map {{ margin: 0; padding: 0; width: 100%; height: 460px; border-radius: 6px; }}
  .cad-tooltip {{
    background: rgba(15, 23, 42, 0.9); color: #fff; border: 1px solid #DC2626;
    font-family: sans-serif; font-size: 12px; font-weight: 600; padding: 3px 7px;
  }}
</style>
</head>
<body>
<div id="map"></div>
<script>
  const map = L.map('map', {{ maxZoom: 19 }}).setView([{center_lat}, {center_lon}], 18);
  const osmLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxNativeZoom: 19,
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap | ППК Росреестр'
  }});
  const satLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
    maxNativeZoom: 18,
    maxZoom: 19,
    attribution: 'Esri Satellite | ППК Росреестр'
  }});
  satLayer.addTo(map);
  osmLayer.setOpacity(0.35).addTo(map);
  L.control.layers(
    {{ "Спутник + Дома (Гибрид)": satLayer, "Карта домов и улиц (OSM)": osmLayer }},
    {{}},
    {{ position: 'topright' }}
  ).addTo(map);
  const polyCoords = {poly_json};
  if (polyCoords.length >= 3) {{
    const poly = L.polygon(polyCoords, {{
      color: '{stroke_color}', weight: 3, fillColor: '{stroke_color}', fillOpacity: 0.16
    }}).addTo(map);
    poly.bindTooltip('Росреестр: {cad_label}', {{permanent: true, className: 'cad-tooltip'}});
    map.fitBounds(poly.getBounds(), {{ maxZoom: 18, padding: [40, 40] }});
  }} else {{
    map.setView([{center_lat}, {center_lon}], 18);
  }}
  const markers = {markers_json};
  markers.forEach((m, i) => {{
    L.circleMarker([m.lat, m.lon], {{
      radius: 6, color: '#0F172A', weight: 2, fillColor: '#2563EB', fillOpacity: 0.95
    }}).addTo(map).bindPopup(m.label);
  }});
</script>
</body>
</html>"""


def main() -> None:
    apply_custom_css()
    plot_ids = get_available_plot_ids()

    # Process any pending plot selection BEFORE widgets are created
    if "pending_plot_id" in st.session_state:
        pending_id = st.session_state.pop("pending_plot_id")
        if pending_id in plot_ids:
            st.session_state["selected_plot_id"] = pending_id
        elif pending_id is None and "selected_plot_id" in st.session_state:
            del st.session_state["selected_plot_id"]

    if "inspected_plots" not in st.session_state:
        st.session_state["inspected_plots"] = {}

    gigachat_probe = GigaChatInspectorService()

    # Top System Header
    _render_html(
        render_sber_header(
            active_objects_count=len(plot_ids),
            rtk_status="FIX Q=16",
            giga_status="GigaChat API Онлайн" if gigachat_probe.is_online else "GigaChat Оффлайн",
            giga_online=gigachat_probe.is_online,
        )
    )

    # Sidebar: Object Selection, Upload & Deletion, Overlay Settings
    with st.sidebar:
        st.markdown("### Объекты мониторинга")

        if plot_ids:
            if (
                "selected_plot_id" not in st.session_state
                or st.session_state["selected_plot_id"] not in plot_ids
            ):
                st.session_state["selected_plot_id"] = plot_ids[0]

            plot_labels = {}
            for pid in plot_ids:
                cred = CreditRegistry.get_credit_object(pid)
                cad_str = cred.cadastral_id if cred else normalize_cadastral_id(pid)
                is_done = pid in st.session_state["inspected_plots"]
                status_tag = "[Проверен]" if is_done else ""
                addr_short = (
                    cred.address[:28] + "…"
                    if (cred and len(cred.address) > 28)
                    else (cred.address if cred else "")
                )
                plot_labels[pid] = (
                    f"{cad_str} {('— ' + addr_short) if addr_short else ''} {status_tag}".strip()
                )

            selected_plot_id = st.selectbox(
                "Выберите кадастровый участок:",
                options=plot_ids,
                format_func=lambda x: plot_labels.get(x, x),
                key="selected_plot_id",
            )

            col_del1, col_del2 = st.columns(2)
            with col_del1:
                if st.button("Удалить участок", use_container_width=True):
                    handle_delete_plot(selected_plot_id)
            with col_del2:
                if st.button("Очистить все", use_container_width=True):
                    handle_delete_all_plots()
        else:
            selected_plot_id = None
            st.info("На сервере нет загруженных объектов. Загрузите свой участок ниже.")

        st.markdown("---")
        st.markdown("### Параметры разметки ЗУ")
        show_boundary = st.checkbox("Красная линия границы ЗУ (Росреестр)", value=True)
        show_setback = st.checkbox("Отступ застройки 3.0 м (СП 53.13330)", value=True)
        boundary_width = st.slider("Толщина линий (px)", min_value=3, max_value=12, value=6)

        st.markdown("---")
        with st.expander("Загрузить новый объект (JPG / ZIP)", expanded=not bool(plot_ids)):
            upload_mode = st.radio(
                "Формат загрузки:",
                options=["Файлы съемки (JPG + MRK)", "ZIP-архив участка"],
                horizontal=False,
            )
            if upload_mode == "Файлы съемки (JPG + MRK)":
                new_cad_id = st.text_input(
                    "Кадастровый номер ЗУ:",
                    placeholder="54:19:164603:3847",
                    help="Геометрия и реквизиты участка запрашиваются из Росреестра / НСПД",
                )
                new_files = st.file_uploader(
                    "Снимки БПЛА (.JPG) и телеметрия (.MRK):",
                    type=["jpg", "jpeg", "tif", "tiff", "mrk", "obs", "nav", "txt"],
                    accept_multiple_files=True,
                    key="sb_files_uploader",
                )
                if st.button("Загрузить объект на сервер", type="primary", use_container_width=True):
                    handle_upload_files(new_cad_id, new_files)
            else:
                zip_cad_id = st.text_input(
                    "Кадастровый номер (необязательно):",
                    placeholder="54:07:055001:178",
                    key="sb_zip_cad",
                )
                up_zip = st.file_uploader(
                    "ZIP-архив папки полета:",
                    type=["zip"],
                    key="sb_zip_uploader",
                )
                if st.button("Распаковать ZIP на сервер", type="primary", use_container_width=True):
                    handle_upload_zip(up_zip, target_plot_id=zip_cad_id)

        with st.expander("Реестр объектов (.xlsx / .csv)", expanded=False):
            up_reg = st.file_uploader(
                "Обновить таблицу реестра:",
                type=["xlsx", "csv"],
                key="sb_reg_uploader",
            )
            if up_reg is not None and st.button("Применить реестр", use_container_width=True):
                try:
                    cnt = CreditRegistry.update_from_file_bytes(up_reg.getvalue(), up_reg.name)
                    st.cache_data.clear()
                    st.toast(f"Загружено записей реестра: {cnt}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Ошибка загрузки реестра: {exc}")

        with st.expander("Настройки GigaChat API", expanded=False):
            st.caption(
                "Статус: Подключен ключ Сбер API (.env)"
                if gigachat_probe.is_online
                else "Статус: Автономный режим"
            )
            gigachat_key = st.text_input(
                "Переопределить ключ GigaChat API:",
                type="password",
                help="По умолчанию используется GIGACHAT_CREDENTIALS из .env",
            )

    # Empty state when all plots are deleted by user
    if not selected_plot_id:
        _render_html(
            """
            <div class="ws-panel">
                <div class="ws-panel-header">Загрузка пользовательских объектов ИЖС</div>
                <p style="color:#334155; font-size:0.95rem; margin-bottom:8px;">
                    Сервер готов к приему данных. Укажите кадастровый номер земельного участка и загрузите
                    снимки с дрона (<code>DJI_*.JPG</code>, <code>*.MRK</code>) или ZIP-архив через панель слева.
                </p>
                <p style="color:#64748B; font-size:0.85rem; margin:0;">
                    При загрузке система автоматически запрашивает координаты поворотных точек, площадь,
                    адрес и ВРИ участка в базе ППК Росреестр / ФГИС ЕЦП НСПД и проецирует красную линию
                    границы на все аэрофотоснимки.
                </p>
            </div>
            """
        )
        return

    # Run inspection for the active plot on demand
    with st.spinner(f"Анализ участка {normalize_cadastral_id(selected_plot_id)}..."):
        cv_result, credit_obj, decision_result = analyze_single_plot(selected_plot_id)
        st.session_state["inspected_plots"][selected_plot_id] = (
            cv_result,
            credit_obj,
            decision_result,
        )

    ds = DroneDataset()
    plot = ds.get_plot(selected_plot_id)
    parcel = CadastralRegistry.get_parcel(
        selected_plot_id, fetch_online=True, plot_dir=plot.dir_path
    )
    paired_shots = match_shots_to_metadata(plot.rgb_images, cv_result.shots_metadata)

    # Top KPI Strip (4 strict corporate cards)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _render_html(
            render_kpi_card(
                title="Кадастровый номер",
                value=cv_result.cadastral_id,
                subtext=(credit_obj.address if credit_obj else (parcel.address if parcel else "Н/Д")),
                icon_type="satellite",
                is_success=cv_result.geocheck.is_geofence_valid,
                is_danger=not cv_result.geocheck.is_geofence_valid,
            )
        )
    with k2:
        _render_html(
            render_kpi_card(
                title="Стадия и готовность (CV)",
                value=f"{cv_result.readiness_pct}% (Стадия {cv_result.readiness_stage}%)",
                subtext=f"{cv_result.stage_title} • {cv_result.detected_material.value}",
                icon_type="chart",
                is_success=(cv_result.readiness_stage == 100),
                is_warning=(cv_result.readiness_stage == 50),
            )
        )
    with k3:
        _render_html(
            render_kpi_card(
                title="Сроки строительства",
                value=f"План: {credit_obj.planned_completion_date if credit_obj else 'Нет'}",
                subtext=(
                    f"Начало: {credit_obj.construction_start_date if credit_obj else 'Нет'} • "
                    f"Факт: {credit_obj.actual_completion_date if credit_obj else 'Нет'}"
                ),
                icon_type="drone",
                is_success=True,
            )
        )
    with k4:
        has_pz_str = credit_obj.has_pz if credit_obj else "Нет"
        _render_html(
            render_kpi_card(
                title="ПЗ / Регистрация / Залог",
                value=f"ПЗ: {has_pz_str}",
                subtext=(
                    f"ОН зарег.: {credit_obj.is_registered if credit_obj else 'Нет'} • "
                    f"Залог: {credit_obj.is_pledged if credit_obj else 'Нет'}"
                ),
                icon_type="check_circle",
                is_success=(has_pz_str.lower() != "да"),
                is_warning=(has_pz_str.lower() == "да"),
            )
        )

    # Workspace Navigation Tabs (4 clean, non-AI functional tabs)
    tab_photos, tab_geo, tab_act, tab_registry = st.tabs(
        [
            f"1. Аэрофотосъемка и границы ЗУ ({len(paired_shots)} кадр.)",
            "2. Геодезия, PPK и карта Росреестра",
            "3. Акт проверки залога (Приложение 1)",
            f"4. Управление объектами ({len(plot_ids)})",
        ]
    )

    # =========================================================================
    # TAB 1: ALL DRONE PHOTOS WITH RED CADASTRAL BOUNDARY + STRUCTURAL PASSPORT
    # =========================================================================
    with tab_photos:
        col_viewer, col_details = st.columns([1.45, 1.0], gap="medium")

        with col_viewer:
            _render_html(
                f"""
                <div class="ws-panel" style="margin-bottom:8px;">
                    <div class="ws-panel-header" style="margin-bottom:0; border-bottom:none; padding-bottom:0;">
                        <span>Фотограмметрическая проекция границ ЗУ на снимки БПЛА</span>
                        <span>Всего кадров: {len(paired_shots)}</span>
                    </div>
                </div>
                """
            )

            if paired_shots:
                view_mode = st.radio(
                    "Режим просмотра снимков:",
                    options=[
                        "Детальный просмотр выбранного кадра",
                        f"Сетка всех кадров участка (Кадры 1–{len(paired_shots)})",
                    ],
                    horizontal=True,
                    label_visibility="collapsed",
                    key=f"view_mode_{selected_plot_id}",
                )

                if view_mode == "Детальный просмотр выбранного кадра":
                    shot_options = list(range(len(paired_shots)))

                    def _fmt_shot(idx: int) -> str:
                        p_img, m_img = paired_shots[idx]
                        angle_str = m_img.shot_angle.value if m_img else "Обзорный"
                        pitch_str = (
                            f"pitch {m_img.gimbal_pitch_deg:.0f}°"
                            if (m_img and m_img.gimbal_pitch_deg is not None)
                            else ""
                        )
                        return f"Кадр {idx + 1} из {len(paired_shots)} — {angle_str} ({p_img.name} {pitch_str})"

                    sel_idx = st.selectbox(
                        "Выберите кадр для инспекции:",
                        options=shot_options,
                        format_func=_fmt_shot,
                        key=f"shot_idx_{selected_plot_id}",
                    )
                    img_path, meta = paired_shots[sel_idx]

                    if show_boundary:
                        meta_dict = meta.model_dump() if meta is not None else None
                        img_bytes = render_cached_boundary_overlay(
                            str(img_path),
                            meta_dict,
                            selected_plot_id,
                            boundary_width,
                            show_setback=show_setback,
                        )
                        st.image(img_bytes, use_container_width=True)
                    else:
                        st.image(str(img_path), use_container_width=True)

                    if meta:
                        rtk_txt = f"RTK {meta.rtk_status}" if meta.rtk_status else "GNSS EXIF"
                        st.caption(
                            f"Файл: {img_path.name} | Ракурс: {meta.shot_angle.value} | "
                            f"Координаты: {meta.latitude:.6f}, {meta.longitude:.6f} | "
                            f"Pitch: {meta.gimbal_pitch_deg}° | Yaw: {meta.gimbal_yaw_deg}° | {rtk_txt}"
                        )
                else:
                    # Grid of ALL drone shots (Shots 1..N) with red boundary overlay!
                    grid_cols = st.columns(2)
                    for idx, (img_path, meta) in enumerate(paired_shots):
                        with grid_cols[idx % 2]:
                            angle_lbl = meta.shot_angle.value if meta else "Кадр БПЛА"
                            st.markdown(
                                f"**Кадр {idx + 1}: {angle_lbl}** (`{img_path.name}`)"
                            )
                            if show_boundary:
                                meta_dict = meta.model_dump() if meta is not None else None
                                img_bytes = render_cached_boundary_overlay(
                                    str(img_path),
                                    meta_dict,
                                    selected_plot_id,
                                    boundary_width,
                                    show_setback=show_setback,
                                )
                                st.image(img_bytes, use_container_width=True)
                            else:
                                st.image(str(img_path), use_container_width=True)
            else:
                st.warning("В папке участка отсутствуют изображения .JPG")

        with col_details:
            # 1. Constructive Readiness Breakdown
            se = cv_result.structural_elements

            def _elem_badge(ok: bool, conf: float) -> str:
                cls_name = "ws-status-ok" if ok else "ws-status-warn"
                txt = f"Выполнено ({int(conf * 100)}%)" if ok else "Отсутствует (0%)"
                return f'<span class="ws-status-pill {cls_name}">{txt}</span>'

            _render_html(
                f"""
                <div class="ws-panel">
                    <div class="ws-panel-header">
                        <span>Оценка строительной готовности</span>
                        <span>Итог: {cv_result.readiness_pct}%</span>
                    </div>
                    <div style="font-size:0.9rem; font-weight:700; color:#0F172A; margin-bottom:10px;">
                        {cv_result.stage_title}
                    </div>
                    <table class="sber-table-v2">
                        <thead>
                            <tr>
                                <th>Конструктивный элемент</th>
                                <th>Вес</th>
                                <th>Статус верификации</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>1. Фундамент (нулевой цикл)</td>
                                <td>15%</td>
                                <td>{_elem_badge(se.has_foundation, se.foundation_confidence)}</td>
                            </tr>
                            <tr>
                                <td>2. Стеновой комплект / каркас</td>
                                <td>40%</td>
                                <td>{_elem_badge(se.has_walls, se.walls_confidence)}</td>
                            </tr>
                            <tr>
                                <td>3. Кровельное покрытие</td>
                                <td>30%</td>
                                <td>{_elem_badge(se.has_roof, se.roof_confidence)}</td>
                            </tr>
                            <tr>
                                <td>4. Тепловой контур (окна, фасад)</td>
                                <td>15%</td>
                                <td>{_elem_badge(se.has_windows_doors, se.windows_doors_confidence)}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """
            )

            # 2. Customer Registry Object Data (strictly matching CSV columns)
            _render_html(
                f"""
                <div class="ws-panel">
                    <div class="ws-panel-header">
                        <span>Данные по объекту из реестра</span>
                        <span>{cv_result.cadastral_id}</span>
                    </div>
                    <div class="ws-kv-grid">
                        <div class="ws-kv-label">Территория:</div>
                        <div class="ws-kv-val">{credit_obj.address if credit_obj else (parcel.address if parcel else "Н/Д")}</div>
                        <div class="ws-kv-label">Кадастровый номер:</div>
                        <div class="ws-kv-val">{cv_result.cadastral_id}</div>
                        <div class="ws-kv-label">Дата начала стр-ва:</div>
                        <div class="ws-kv-val">{credit_obj.construction_start_date if credit_obj else "Нет"}</div>
                        <div class="ws-kv-label">Плановая дата:</div>
                        <div class="ws-kv-val">{credit_obj.planned_completion_date if credit_obj else "Нет"}</div>
                        <div class="ws-kv-label">Фактическая дата:</div>
                        <div class="ws-kv-val">{credit_obj.actual_completion_date if credit_obj else "Нет"}</div>
                        <div class="ws-kv-label">Наличие ПЗ:</div>
                        <div class="ws-kv-val">{credit_obj.has_pz if credit_obj else "Нет"}</div>
                        <div class="ws-kv-label">ОН зарегистрирован:</div>
                        <div class="ws-kv-val">{credit_obj.is_registered if credit_obj else "Нет"}</div>
                        <div class="ws-kv-label">Залог:</div>
                        <div class="ws-kv-val">{credit_obj.is_pledged if credit_obj else "Нет"}</div>
                        <div class="ws-kv-label">Наличие фото с неба:</div>
                        <div class="ws-kv-val">{credit_obj.has_aerial_photo if credit_obj else "Да"}</div>
                    </div>
                </div>
                """
            )

            # 3. Decision Engine Action Box
            steps_html = "".join(f"<li>{s}</li>" for s in decision_result.reasoning)
            _render_html(
                f"""
                <div class="ws-panel">
                    <div class="ws-panel-header">
                        <span>Решение по кредитному досье</span>
                        <span>{decision_result.strategy_code}</span>
                    </div>
                    <div style="font-weight:700; color:#0F172A; margin-bottom:6px;">
                        {decision_result.strategy.value}
                    </div>
                    <div style="font-size:0.84rem; color:#334155; margin-bottom:8px;">
                        {decision_result.action_required}
                    </div>
                    <div style="font-size:0.82rem; color:#475569;">
                        <strong>Обоснование решения (АС «Калита» — {decision_result.kalita_status}):</strong>
                        <ul style="margin:4px 0 0 18px; padding:0;">{steps_html}</ul>
                    </div>
                </div>
                """
            )

            # 4. Sber AI Agent (GigaChat-2-Max Vision) Findings
            if cv_result.operator_notes or cv_result.defects_detected:
                defects_li = "".join(f"<li>{d}</li>" for d in cv_result.defects_detected)
                defects_block = (
                    f'<div style="margin-top:8px; font-size:0.82rem; color:#92400E;">'
                    f'<strong>Выявленные помехи съемке и замечания ИИ-агента:</strong>'
                    f'<ul style="margin:4px 0 0 18px; padding:0;">{defects_li}</ul></div>'
                    if defects_li
                    else ""
                )
                notes_html = (
                    "<br><br>".join(cv_result.operator_notes)
                    if isinstance(cv_result.operator_notes, list)
                    else str(cv_result.operator_notes or "")
                )
                _render_html(
                    f"""
                    <div class="ws-panel">
                        <div class="ws-panel-header">
                            <span>Заключение ИИ-агента Сбербанка (GigaChat-2-Max Vision)</span>
                            <span>Мультимодальный анализ</span>
                        </div>
                        <div style="font-size:0.85rem; color:#1E293B; line-height:1.5;">
                            {notes_html or "Анализ выполнен в штатном режиме."}
                        </div>
                        {defects_block}
                    </div>
                    """
                )

    # =========================================================================
    # TAB 2: GEODESY, PPK & ROSREESTR MAP
    # =========================================================================
    with tab_geo:
        g_col1, g_col2 = st.columns([1.4, 1.0], gap="medium")
        with g_col1:
            _render_html(
                f"""
                <div class="ws-panel" style="margin-bottom:8px;">
                    <div class="ws-panel-header" style="margin-bottom:0; border-bottom:none; padding-bottom:0;">
                        <span>Карта кадастрового контура ЗУ и точек съемки БПЛА</span>
                        <span>Дистанция до центра: {cv_result.geocheck.distance_to_cadastre_m:.1f} м</span>
                    </div>
                </div>
                """
            )
            fmap_html = build_inspection_folium_map(
                parcel=parcel,
                shots_meta=cv_result.shots_metadata,
                is_geofence_valid=cv_result.geocheck.is_geofence_valid,
            )
            components.html(fmap_html, height=460)

        with g_col2:
            gc = cv_result.geocheck
            comp = cv_result.completeness
            ppk = cv_result.ppk_summary
            ms = cv_result.multispectral

            _render_html(
                f"""
                <div class="ws-panel">
                    <div class="ws-panel-header">
                        <span>Геовалидация и полнота ракурсов</span>
                        <span>{"СООТВЕТСТВУЕТ" if gc.is_geofence_valid else "НАРУШЕНИЕ ГЕОЗОНЫ"}</span>
                    </div>
                    <div class="ws-kv-grid">
                        <div class="ws-kv-label">Статус верификации:</div>
                        <div class="ws-kv-val">{gc.verification_status} ({comp.total_shots} снимков)</div>
                        <div class="ws-kv-label">Отклонение от центра:</div>
                        <div class="ws-kv-val">{gc.distance_to_cadastre_m:.1f} м (прицел: {gc.mean_camera_aim_dist_m:.1f} м)</div>
                        <div class="ws-kv-label">Фасадные ракурсы:</div>
                        <div class="ws-kv-val">{comp.facades_count} из 4 (С, Ю, В, З)</div>
                        <div class="ws-kv-label">Надирные ракурсы:</div>
                        <div class="ws-kv-val">{comp.nadir_count} из 2 (высокий / низкий)</div>
                        <div class="ws-kv-label">RINEX / PPK логи:</div>
                        <div class="ws-kv-val">{ppk.rinex_version if (ppk and ppk.has_ppk) else "RTK MRK"}</div>
                        <div class="ws-kv-label">Индекс NDVI (средн.):</div>
                        <div class="ws-kv-val">{f"{ms.mean_ndvi:.3f} ({ms.site_activity_status})" if (ms and ms.has_multispectral) else "Только RGB"}</div>
                    </div>
                </div>
                """
            )

            if cv_result.shots_metadata:
                rows_tr = ""
                for idx, m in enumerate(cv_result.shots_metadata, start=1):
                    rows_tr += (
                        f"<tr>"
                        f"<td>{idx}</td>"
                        f"<td>{m.filename}</td>"
                        f"<td>{m.shot_angle.value}</td>"
                        f"<td>{m.latitude:.6f}, {m.longitude:.6f}</td>"
                        f"<td>{m.gimbal_pitch_deg}° / {m.gimbal_yaw_deg}°</td>"
                        f"</tr>"
                    )
                _render_html(
                    f"""
                    <div class="ws-panel">
                        <div class="ws-panel-header">Журнал телеметрии снимков БПЛА</div>
                        <table class="sber-table-v2">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Файл</th>
                                    <th>Ракурс</th>
                                    <th>WGS-84 (Lat, Lon)</th>
                                    <th>Pitch / Yaw</th>
                                </tr>
                            </thead>
                            <tbody>{rows_tr}</tbody>
                        </table>
                    </div>
                    """
                )

    # =========================================================================
    # TAB 3: COLLATERAL INSPECTION ACT (APPENDIX 1) & EXPERT SUMMARY
    # =========================================================================
    with tab_act:
        gigachat_srv = GigaChatInspectorService(credentials=gigachat_key if gigachat_key else None)
        ai_summary = gigachat_srv.generate_summary_for_ui(
            cv_result=cv_result,
            credit_obj=credit_obj,
            decision=decision_result,
            plot_dir=ds.root_dir / selected_plot_id,
            parcel=parcel,
        )

        _render_html(
            f"""
            <div class="ws-panel">
                <div class="ws-panel-header">
                    <span>Экспертное заключение по объекту {cv_result.cadastral_id}</span>
                    <span>{"GigaChat API Сбер" if gigachat_srv.is_online else "АС «Калита»"}</span>
                </div>
                <div style="font-size:0.9rem; color:#1E293B; line-height:1.55;">
                    {ai_summary}
                </div>
            </div>
            """
        )

        act_photos: list[ReportPhotoItem] = []
        for idx, (p_shot, m_shot) in enumerate(paired_shots[:4], start=1):
            b64_uri = get_base64_image(p_shot, apply_overlay=True, meta=m_shot, parcel=parcel)
            if b64_uri:
                angle_str = m_shot.shot_angle.value if m_shot else "Обзорный"
                act_photos.append(
                    ReportPhotoItem(
                        file_path=b64_uri,
                        caption=f"Кадр {idx} ({angle_str}): {p_shot.name} — Красная линия ЗУ Росреестр",
                        is_key_view=(idx == 1),
                    )
                )

        act_data = build_act_from_cv(
            cv_result=cv_result,
            credit_obj=credit_obj,
            decision=decision_result,
            gigachat_summary=ai_summary,
            photos=act_photos,
        )
        act_gen = ActGenerator()
        html_report = act_gen.render_html(act_data)

        st.download_button(
            label="Скачать официальный Акт проверки залога (Приложение 1, HTML / Печать в PDF)",
            data=html_report.encode("utf-8"),
            file_name=f"Act_Collateral_{selected_plot_id}.html",
            mime="text/html",
            type="primary",
            use_container_width=True,
            key=f"download_act_{selected_plot_id}",
        )

        components.html(html_report, height=760, scrolling=True)

    # =========================================================================
    # TAB 4: USER PARCEL UPLOAD, DELETION & PORTFOLIO REGISTRY
    # =========================================================================
    with tab_registry:
        u_col1, u_col2 = st.columns(2, gap="medium")
        with u_col1:
            _render_html(
                """
                <div class="ws-panel" style="margin-bottom:8px;">
                    <div class="ws-panel-header" style="margin-bottom:0; border-bottom:none; padding-bottom:0;">
                        <span>Добавить новый дом / участок (Файлы БПЛА + Кадастровый номер)</span>
                    </div>
                </div>
                """
            )
            form_cad = st.text_input(
                "Кадастровый номер земельного участка:",
                placeholder="Например: 54:19:164603:3847",
                key="main_upload_cad",
            )
            form_files = st.file_uploader(
                "Выберите снимки с дрона (.JPG) и навигационный файл (.MRK):",
                type=["jpg", "jpeg", "tif", "tiff", "mrk", "obs", "nav", "txt"],
                accept_multiple_files=True,
                key="main_upload_files",
            )
            if st.button("Загрузить участок и запросить данные Росреестра", type="primary"):
                handle_upload_files(form_cad, form_files)

        with u_col2:
            _render_html(
                """
                <div class="ws-panel" style="margin-bottom:8px;">
                    <div class="ws-panel-header" style="margin-bottom:0; border-bottom:none; padding-bottom:0;">
                        <span>Пакетная загрузка ZIP-архива и управление реестром</span>
                    </div>
                </div>
                """
            )
            form_zip_cad = st.text_input(
                "Целевой кадастровый номер (для плоского архива):",
                placeholder="Необязательно, если папки названы по кадастру",
                key="main_zip_cad",
            )
            form_zip = st.file_uploader(
                "Выберите ZIP-архив:",
                type=["zip"],
                key="main_zip_file",
            )
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                if st.button("Загрузить ZIP-архив", type="primary", use_container_width=True):
                    handle_upload_zip(form_zip, target_plot_id=form_zip_cad)
            with b_col2:
                if st.button("Проверить все участки пакетно", use_container_width=True):
                    with st.spinner("Пакетный расчет всех участков..."):
                        for pid in plot_ids:
                            if pid not in st.session_state["inspected_plots"]:
                                st.session_state["inspected_plots"][pid] = analyze_single_plot(pid)
                    st.rerun()

        # Table of all plots on the server with direct Delete / Inspect buttons
        portfolio_data = get_portfolio_summary(plot_ids, st.session_state["inspected_plots"])
        _render_html(
            f"""
            <div class="ws-panel" style="margin-top:12px; margin-bottom:8px;">
                <div class="ws-panel-header" style="margin-bottom:0; border-bottom:none; padding-bottom:0;">
                    <span>Реестр загруженных объектов на сервере ({len(portfolio_data)})</span>
                    <span>Проверено: {len(st.session_state["inspected_plots"])} из {len(portfolio_data)}</span>
                </div>
            </div>
            """
        )

        for item in portfolio_data:
            raw_pid = item["raw_id"]
            p_stats = ParcelManager.get_plot_stats(raw_pid)
            r_c1, r_c2, r_c3, r_c4, r_c5 = st.columns([2.0, 2.7, 2.0, 1.2, 1.0])
            with r_c1:
                st.markdown(f"**{item['cadastral_id']}**")
                st.caption(f"Папка: `{raw_pid}` ({p_stats['rgb_count']} фото, {p_stats['size_mb']} МБ)")
            with r_c2:
                st.markdown(f"{item['address']}")
                st.caption(
                    f"Начало: {item['start_date']} • План: {item['planned_date']} • Факт: {item['actual_date']}"
                )
            with r_c3:
                st.caption(
                    f"ПЗ: {item['has_pz']} | ОН зарег.: {item['is_registered']} | Залог: {item['is_pledged']} | Фото: {item['has_aerial_photo']}"
                )
                if item["is_inspected"]:
                    st.markdown(
                        f"Готовность: **{item['readiness_pct']}%** (Стадия {item['stage']}%)"
                    )
                else:
                    st.markdown("Статус: *Ожидает осмотра*")
            with r_c4:
                if st.button("Открыть", key=f"open_{raw_pid}", use_container_width=True):
                    st.session_state["pending_plot_id"] = raw_pid
                    st.rerun()
            with r_c5:
                if st.button("Удалить", key=f"del_{raw_pid}", use_container_width=True):
                    handle_delete_plot(raw_pid)
            _render_html("<hr style='margin:6px 0; border-color:#F1F5F9;'>")


if __name__ == "__main__":
    main()
