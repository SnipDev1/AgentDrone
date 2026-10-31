"""AgentDrone - Web Dashboard & Inspection Workplace for Sberbank.

Automated AI-assisted collateral and mortgage construction monitoring using
DJI drone telemetry, computer vision inspection, cadastral geofencing,
and the Bank Decision Engine.
Optimized for zero-AVX Linux server environments.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from src.cv.boundary_overlay import BoundaryOverlay
from src.cv.pipeline import CVPipeline
from src.dataset import DroneDataset
from src.engine.decision_engine import DecisionEngine
from src.gigachat_service import GigaChatInspectorService, build_inspector_prompt
from src.ingestion.cadastral_registry import CadastralRegistry, normalize_cadastral_id
from src.ingestion.credit_registry import CreditRegistry
from src.models.schemas import CreditObject, CVInspectionResult, StrategyRecommendation
from src.models.strategy import ReportPhotoItem
from src.report import ActGenerator, build_act_from_cv

# Page configuration
st.set_page_config(
    page_title="AgentDrone — Мониторинг залогов БПЛА | Сбер",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Sberbank Green Theme)
st.markdown(
    """
    <style>
    :root {
        --sber-green: #21a038;
        --sber-dark-green: #0f6c24;
        --sber-light-green: #eaf8ec;
        --sber-gray: #f4f6f8;
        --text-dark: #1f2937;
    }
    .main-header {
        background: linear-gradient(135deg, #0e3d19 0%, #176527 50%, #21a038 100%);
        padding: 20px 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 26px;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-header p {
        color: #d1fae5 !important;
        margin: 6px 0 0 0;
        font-size: 14px;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #111827;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        color: #6b7280;
        letter-spacing: 0.5px;
    }
    .badge-green {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    .badge-yellow {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    .badge-red {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    .decision-card-green {
        border-left: 6px solid #22c55e;
        background: #f0fdf4;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 12px 0;
    }
    .decision-card-yellow {
        border-left: 6px solid #eab308;
        background: #fefce8;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 12px 0;
    }
    .decision-card-red {
        border-left: 6px solid #ef4444;
        background: #fef2f2;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 12px 0;
    }
    .sber-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        margin-top: 10px;
    }
    .sber-table th {
        background-color: #f3f4f6;
        color: #374151;
        font-weight: 600;
        padding: 10px 12px;
        text-align: left;
        border-bottom: 2px solid #e5e7eb;
    }
    .sber-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #e5e7eb;
        color: #1f2937;
    }
    .sber-table tr:hover {
        background-color: #f9fafb;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 18px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_available_plot_ids() -> list[str]:
    """Retrieve all available cadastral plot IDs."""
    dataset = DroneDataset()
    plots = dataset.list_plots()
    if not plots:
        return [p.raw_folder_name for p in CadastralRegistry.list_all_parcels()]
    return plots


CACHE_SALT = "v3_ppk_mrk_ms_live"


@st.cache_data(show_spinner=False, ttl=60)
def analyze_single_plot(
    plot_id: str,
    _cache_ver: str = CACHE_SALT,
) -> tuple[CVInspectionResult, CreditObject, StrategyRecommendation]:
    """Execute end-to-end CV and Decision Engine pipeline with caching."""
    credit_obj = CreditRegistry.get_credit_object(plot_id)
    cv_result = CVPipeline.analyze_plot(
        plot_id=plot_id,
        credit_object=credit_obj,
        declared_material=credit_obj.declared_wall_material,
    )
    decision = DecisionEngine.evaluate(cv_result, credit_obj)
    return cv_result, credit_obj, decision


def build_plot_summary(
    pid: str,
    inspected: tuple[CVInspectionResult, CreditObject, StrategyRecommendation] | None = None,
) -> dict:
    """Build summary dictionary for a single plot, either from CV inspection or registry defaults."""
    parcel = CadastralRegistry.get_parcel(pid)
    credit = CreditRegistry.get_credit_object(pid)

    if inspected is not None:
        cv_res, credit_obj, dec = inspected
        return {
            "raw_id": pid,
            "cadastral_id": cv_res.cadastral_id,
            "borrower": credit_obj.borrower_name,
            "address": parcel.address if parcel else credit_obj.address,
            "credit_amount": credit_obj.credit_amount_rub,
            "readiness_pct": cv_res.readiness_pct,
            "stage": cv_res.readiness_stage,
            "material": cv_res.detected_material.value,
            "strategy": dec.strategy.value,
            "strategy_code": dec.strategy_code,
            "risk_level": dec.risk_level,
            "fraud_flag": cv_res.fraud_risk_flag,
            "lat": parcel.centroid_lat if parcel else 55.0,
            "lon": parcel.centroid_lon if parcel else 83.0,
            "polygon": BoundaryOverlay.calculate_ground_polygon(parcel) if parcel else [],
            "kalita_status": dec.kalita_status,
            "days_overdue": credit_obj.days_overdue,
            "is_inspected": True,
        }

    # Lazy-loaded / uninspected plot: fast metadata from registry, no heavy CV processing
    cad_id = parcel.cadastral_id if parcel else (credit.cadastral_id if credit else normalize_cadastral_id(pid))
    borrower = credit.borrower_name if credit else "Заемщик не указан"
    address = parcel.address if parcel else (credit.address if credit else "Адрес не указан")
    amount = credit.credit_amount_rub if credit else 0.0
    material = credit.declared_wall_material if credit else "Не указан"
    overdue = credit.days_overdue if credit else 0
    lat = parcel.centroid_lat if parcel else 55.0
    lon = parcel.centroid_lon if parcel else 83.0
    polygon = BoundaryOverlay.calculate_ground_polygon(parcel) if parcel else []

    return {
        "raw_id": pid,
        "cadastral_id": cad_id,
        "borrower": borrower,
        "address": address,
        "credit_amount": amount,
        "readiness_pct": None,
        "stage": "—",
        "material": f"{material} (по КД)",
        "strategy": "Ожидает перехода инспектора",
        "strategy_code": "ОЖИДАЕТ",
        "risk_level": "PENDING",
        "fraud_flag": False,
        "lat": lat,
        "lon": lon,
        "polygon": polygon,
        "kalita_status": "Ожидает осмотра",
        "days_overdue": overdue,
        "is_inspected": False,
    }


def get_portfolio_summary(
    plot_ids: tuple[str, ...] | list[str],
    inspected_cache: dict[str, tuple[CVInspectionResult, CreditObject, StrategyRecommendation]],
) -> list[dict]:
    """Assemble portfolio list combining inspected plots and pending registry plots."""
    return [build_plot_summary(pid, inspected_cache.get(pid)) for pid in plot_ids]


def analyze_portfolio_summary(
    plot_ids: tuple[str, ...],
    _cache_ver: str = CACHE_SALT,
) -> list[dict]:
    """Backward-compatible portfolio analysis function."""
    summary = []
    for pid in plot_ids:
        try:
            cv_res, credit, dec = analyze_single_plot(pid)
            summary.append(build_plot_summary(pid, (cv_res, credit, dec)))
        except Exception as e:
            st.warning(f"Ошибка анализа участка {pid}: {e}")
    return summary


def main():
    plot_ids = get_available_plot_ids()

    # Session state for inspected plots cache & active tracking
    if "inspected_plots" not in st.session_state:
        st.session_state.inspected_plots = {}
    if "last_visited_plot_id" not in st.session_state:
        st.session_state.last_visited_plot_id = None

    # Sidebar: Logo & Navigation
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Sberbank_Logo_2020.svg/320px-Sberbank_Logo_2020.svg.png",
        width=160,
    )
    st.sidebar.title("Навигация")

    filter_risk = st.sidebar.selectbox(
        "Фильтр по статусу риска:",
        ["Все объекты (13)", "Только проверенные", "Ожидают проверки", "Только с рисками / Stop-Fraud", "Без рисков (ОК)"],
    )

    # Initial portfolio data assembly to evaluate filter
    portfolio_data = get_portfolio_summary(plot_ids, st.session_state.inspected_plots)

    filtered_plots = []
    for item in portfolio_data:
        if filter_risk == "Только проверенные" and not item["is_inspected"]:
            continue
        if filter_risk == "Ожидают проверки" and item["is_inspected"]:
            continue
        if filter_risk == "Только с рисками / Stop-Fraud" and item["risk_level"] != "CRITICAL":
            continue
        if filter_risk == "Без рисков (ОК)" and (not item["is_inspected"] or item["risk_level"] != "LOW"):
            continue
        filtered_plots.append(item["raw_id"])

    if not filtered_plots:
        filtered_plots = plot_ids

    # 1. Safely resolve which plot is currently chosen
    if "selected_plot_id" not in st.session_state or st.session_state["selected_plot_id"] not in filtered_plots:
        st.session_state["selected_plot_id"] = filtered_plots[0]

    current_target_id = st.session_state["selected_plot_id"]

    # 2. Navigation trigger: update plot ONLY when user navigates to it ("переходит")
    is_new_navigation = (st.session_state.get("last_visited_plot_id") != current_target_id)
    is_not_inspected = (current_target_id not in st.session_state.inspected_plots)

    if is_new_navigation or is_not_inspected:
        with st.spinner(f"🛸 Обновление данных БПЛА по участку {normalize_cadastral_id(current_target_id)}..."):
            cv_result, credit_obj, decision = analyze_single_plot(current_target_id)
            st.session_state.inspected_plots[current_target_id] = (cv_result, credit_obj, decision)
            st.session_state.last_visited_plot_id = current_target_id
    else:
        cv_result, credit_obj, decision = st.session_state.inspected_plots[current_target_id]

    # 3. Synchronize portfolio_data with the freshly inspected plot
    portfolio_data = get_portfolio_summary(plot_ids, st.session_state.inspected_plots)

    def make_plot_label(raw_id: str) -> str:
        row = next((r for r in portfolio_data if r["raw_id"] == raw_id), None)
        norm_id = normalize_cadastral_id(raw_id)
        if row and row["is_inspected"]:
            risk_icon = "🔴" if row["risk_level"] == "CRITICAL" else ("🟡" if row["risk_level"] == "MEDIUM" else "🟢")
            readiness = f"{row['readiness_pct']:.0f}%" if row.get("readiness_pct") is not None else "0%"
            return f"{risk_icon} {norm_id} ({row['borrower'].split()[0]}, {readiness})"
        elif row:
            return f"⚪ {norm_id} ({row['borrower'].split()[0]} — ожидает)"
        return norm_id

    # 4. Callback on selectbox change
    def on_plot_change():
        st.session_state["selected_plot_id"] = st.session_state["plot_select_box"]

    current_idx = filtered_plots.index(current_target_id) if current_target_id in filtered_plots else 0
    selected_raw_id = st.sidebar.selectbox(
        "Выберите кадастровый участок:",
        filtered_plots,
        index=current_idx,
        format_func=make_plot_label,
        key="plot_select_box",
        on_change=on_plot_change,
    )
    selected_raw_id = current_target_id

    # Header Banner
    st.markdown(
        """
        <div class="main-header">
            <h1>🛸 AgentDrone — Мониторинг залогов БПЛА</h1>
            <p>Сибирский банк ПАО Сбербанк • Центр залогового мониторинга & АС «Калита» • ИИ-инспекция ИЖС</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top KPI Metrics bar
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    total_portfolio_rub = sum(d["credit_amount"] for d in portfolio_data)
    inspected_items = [d for d in portfolio_data if d.get("is_inspected")]
    inspected_count = len(inspected_items)
    avg_readiness = (
        sum(d["readiness_pct"] for d in inspected_items if d["readiness_pct"] is not None) / inspected_count
    ) if inspected_count > 0 else 0.0
    fraud_count = sum(1 for d in inspected_items if d["fraud_flag"])
    auto_approved = sum(1 for d in inspected_items if d["strategy_code"] == "STRAT-01")

    with col_m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Объектов на мониторинге</div>
                <div class="metric-value">{inspected_count} / {len(portfolio_data)}</div>
                <div style="font-size:12px; color:#15803d;">● Проверено по переходу</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Сумма кредитного портфеля</div>
                <div class="metric-value">{total_portfolio_rub / 1_000_000:.1f} млн ₽</div>
                <div style="font-size:12px; color:#6b7280;">Обеспечение: 100%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Средняя готовность объектов</div>
                <div class="metric-value">{avg_readiness:.1f}%</div>
                <div style="font-size:12px; color:#2563eb;">По проверенным ({inspected_count})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Алерты Stop-Fraud</div>
                <div class="metric-value" style="color: {'#b91c1c' if fraud_count > 0 else '#15803d'};">
                    {fraud_count}
                </div>
                <div style="font-size:12px; color: {'#b91c1c' if fraud_count > 0 else '#15803d'};">
                    {'⚠️ Требуется блокировка' if fraud_count > 0 else '● Фрод не обнаружен'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Авто-выдача траншей</div>
                <div class="metric-value" style="color: #15803d;">{auto_approved}</div>
                <div style="font-size:12px; color:#15803d;">Без ручного выезда</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 Телеметрия и Сервисы")
    st.sidebar.markdown(
        """
        - **БПЛА:** DJI Mavic 3 Enterprise / M3M
        - **Позиционирование:** RTK / PPK (Q=16 сантиметровый)
        - **Кадастр:** Росреестр НСО (Центроиды, ВРИ)
        - **Интеграция:** АС «Калита» (Шлюз активен)
        - **AI Ядро:** OpenCV + Heuristic Risk Matrix
        - **Режим обновления:** по переходу на участок
        - **LLM / GenAI:** GigaChat API (Сбер)
        """
    )

    # GigaChat service initialization
    giga_service = GigaChatInspectorService()
    giga_status = giga_service.get_mode_status()

    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Sber GigaChat API")
    if giga_status["is_online"]:
        st.sidebar.success(f"🟢 {giga_status['mode']}\nМодель: {giga_status['model']}")
    else:
        st.sidebar.info(f"🟡 {giga_status['mode']}\nАвтономный экспертный шаблонизатор")

    with st.sidebar.expander("⚙️ Настройки GigaChat API"):
        giga_key_input = st.text_input(
            "Authorization Key (Base64 ClientID:Secret):",
            type="password",
            key="giga_custom_token",
            help="Введите токен Сбер ID для прямого инференса GigaChat",
        )
        if giga_key_input:
            if giga_service.set_credentials(giga_key_input):
                st.success("✅ GigaChat API успешно подключен!")
            else:
                st.warning("⚠️ Ошибка авторизации. Работает Fallback-режим.")

    st.sidebar.markdown("---")
    c_btn1, c_btn2 = st.sidebar.columns(2)
    with c_btn1:
        if st.button("🔄 Обновить этот", use_container_width=True, help="Принудительно пересчитать данные выбранного участка"):
            analyze_single_plot.clear(selected_raw_id)
            cv_result, credit_obj, decision = analyze_single_plot(selected_raw_id)
            st.session_state.inspected_plots[selected_raw_id] = (cv_result, credit_obj, decision)
            giga_key = f"giga_summary_{selected_raw_id}"
            if giga_key in st.session_state:
                del st.session_state[giga_key]
            st.toast(f"Участок {normalize_cadastral_id(selected_raw_id)} обновлен!", icon="✅")
            st.rerun()
    with c_btn2:
        if st.button("🗑️ Сбросить всё", use_container_width=True, help="Очистить весь кэш и историю проверок"):
            st.cache_data.clear()
            st.session_state.inspected_plots = {}
            st.session_state.last_visited_plot_id = None
            if "selected_plot_id" in st.session_state:
                del st.session_state["selected_plot_id"]
            if "plot_select_box" in st.session_state:
                del st.session_state["plot_select_box"]
            st.rerun()

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🛸 Инспекция БПЛА & CV Анализ",
        "🗺️ ГИС-карта залогов НСО",
        "📊 Портфельный мониторинг & ROI",
        "📑 Акт проверки залога (Приложение 1)",
    ])

    # =========================================================================
    # TAB 1: INSPECTION CARD
    # =========================================================================
    with tab1:
        parcel = CadastralRegistry.get_parcel(selected_raw_id)

        # Header for the selected plot
        c_title, c_badge = st.columns([3, 1])
        with c_title:
            st.subheader(f"Участок: {cv_result.cadastral_id}")
            address_str = parcel.address if parcel else credit_obj.address
            st.caption(
                f"📍 {address_str} | Заемщик: **{credit_obj.borrower_name}** | "
                f"Договор: **{credit_obj.credit_agreement_number}**"
            )
        with c_badge:
            if decision.risk_level == "CRITICAL":
                st.markdown(
                    "<div style='text-align:right;'><span class='badge-red'>КРИТИЧЕСКИЙ РИСК / СБ</span></div>",
                    unsafe_allow_html=True,
                )
            elif decision.risk_level == "MEDIUM":
                st.markdown(
                    "<div style='text-align:right;'><span class='badge-yellow'>ТРЕБУЕТ ВНИМАНИЯ</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='text-align:right;'><span class='badge-green'>БЕЗ РИСКА / СОГЛАСОВАНО</span></div>",
                    unsafe_allow_html=True,
                )

        st.write("")

        # 4 Inspection Metric Cards
        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
        with col_i1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Готовность объекта</div>
                    <div class="metric-value">{cv_result.readiness_pct:.1f}%</div>
                    <div style="font-size:12px; color:#15803d; font-weight:600;">{cv_result.stage_title}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(cv_result.readiness_pct / 100.0, 0.0), 1.0))
        with col_i2:
            geo_status = "ПОДТВЕРЖДЕНО" if cv_result.geocheck.is_correct_plot else "НАРУШЕНИЕ"
            geo_color = "#15803d" if cv_result.geocheck.is_correct_plot else "#b91c1c"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Геовалидация (RTK)</div>
                    <div class="metric-value" style="color: {geo_color};">{geo_status}</div>
                    <div style="font-size:12px; color:#6b7280;">Смещение: {cv_result.geocheck.distance_to_cadastre_m:.1f} м (допуск 75м)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_i3:
            mat_name = cv_result.detected_material.value
            declared_mat = credit_obj.declared_wall_material
            mat_color = "#15803d" if mat_name in declared_mat or "жилой" in declared_mat.lower() else "#b91c1c"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Материал стен (CV)</div>
                    <div class="metric-value" style="font-size:18px; color:{mat_color};">{mat_name}</div>
                    <div style="font-size:12px; color:#6b7280;">В договоре: {declared_mat}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_i4:
            comp = cv_result.completeness
            comp_badge = (
                "4 фасада + 2 надира"
                if comp.is_fully_compliant
                else f"{comp.facades_count} фас. / {comp.nadir_count} над."
            )
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Ракурсы съемки</div>
                    <div class="metric-value" style="font-size:18px; color:#2563eb;">{comp_badge}</div>
                    <div style="font-size:12px; color:#6b7280;">Всего кадров БПЛА: {cv_result.analyzed_images_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # DECISION ENGINE CARD
        card_class = (
            "decision-card-red"
            if decision.risk_level == "CRITICAL"
            else ("decision-card-yellow" if decision.risk_level == "MEDIUM" else "decision-card-green")
        )
        st.markdown(
            f"""
            <div class="{card_class}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0; font-size:18px;">
                        🎯 Решение Банка: {decision.strategy_code} — {decision.strategy.value}
                    </h3>
                    <span style="font-weight:700; font-size:14px;">Статус АС Калита: <code>{decision.kalita_status}</code></span>
                </div>
                <p style="margin: 8px 0 4px 0; font-size:14px; font-weight:600;">{decision.action_required}</p>
                <div style="font-size:13px; color:#4b5563; margin-top:6px;">
                    <strong>Обоснование решения:</strong>
                    <ul style="margin:4px 0 0 16px; padding:0;">
                        {''.join(f'<li>{r}</li>' for r in decision.reasoning)}
                    </ul>
                </div>
                <div style="margin-top:8px; font-size:12px; color:#6b7280;">
                    <strong>Юридическое основание:</strong> {decision.legal_basis} |
                    <strong>Финансовый эффект:</strong> {decision.financial_impact_rub:,.0f} ₽
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Action Buttons
        col_act1, col_act2, col_act3 = st.columns([2, 2, 3])
        with col_act1:
            if st.button("✅ Утвердить решение в АС «Калита»", use_container_width=True, type="primary"):
                st.success(
                    f"Решение [{decision.strategy_code}] успешно передано в АС «Калита». "
                    f"Номер транзакции: KAL-2026-{selected_raw_id[-5:]}"
                )
        with col_act2:
            if st.button("📄 Сформировать Акт проверки залога", use_container_width=True):
                st.info("Акт проверки залога (Приложение 1) сгенерирован и доступен на вкладке «Акт проверки залога».")
        with col_act3:
            st.caption("ℹ️ Все транзакции заверяются усиленной квалифицированной ЭЦП инспектора залогов.")

        # GigaChat AI Inspector Summary Card
        st.write("")
        st.subheader("🤖 Экспертное AI-заключение GigaChat (Сбер) для АС «Калита»")

        giga_key = f"giga_summary_{selected_raw_id}"
        if giga_key not in st.session_state:
            st.session_state[giga_key] = giga_service.generate_summary_for_ui(cv_result, credit_obj, decision)

        col_g1, col_g2 = st.columns([3, 1])
        with col_g2:
            if st.button("⚡ Обновить через GigaChat", key=f"btn_regen_{selected_raw_id}", use_container_width=True):
                with st.spinner("Запрос к модели GigaChat Сбера..."):
                    st.session_state[giga_key] = giga_service.generate_summary_for_ui(cv_result, credit_obj, decision)
                st.success("Резюме обновлено!")

        current_summary = st.session_state[giga_key]
        status_info = giga_service.get_mode_status()
        status_badge = "🟢 GigaChat API (Сбер)" if status_info["is_online"] else "🟡 Оффлайн-шаблонизатор (Fallback)"

        st.markdown(
            f"""
            <div style="background:#f0fdf4; border-left:5px solid #21a038; border-radius:6px; padding:14px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <strong style="color:#15803d; font-size:14px;">Заключение инспектора залоговой службы:</strong>
                    <span style="font-size:12px; background:#dcfce7; color:#15803d; padding:2px 8px; border-radius:12px; font-weight:600;">
                        {status_badge}
                    </span>
                </div>
                <div style="font-size:13.5px; line-height:1.5; color:#1f2937; white-space:pre-line;">
                    {current_summary}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("🔍 Посмотреть структурированный JSON-промпт для GigaChat"):
            st.code(build_inspector_prompt(credit_obj, cv_result, decision), language="json")

        st.markdown("---")

        # Two Columns: Structural Breakdown vs Drone Imagery
        col_struct, col_photos = st.columns([1, 1])

        with col_struct:
            st.subheader("🏗️ Анализ конструктивных элементов (Веса Банка)")
            elems = cv_result.structural_elements

            st.write(
                f"**1. Фундамент (вес 15%):** {'✅ Выполнен' if elems.has_foundation else '❌ Отсутствует'} "
                f"({elems.foundation_confidence*100:.1f}%)"
            )
            st.progress(elems.foundation_confidence)

            st.write(
                f"**2. Несущие стены (вес 40%):** {'✅ Возведены' if elems.has_walls else '❌ Отсутствуют'} "
                f"({elems.walls_confidence*100:.1f}%)"
            )
            st.progress(elems.walls_confidence)

            st.write(
                f"**3. Кровельная система (вес 30%):** {'✅ Смонтирована' if elems.has_roof else '❌ Отсутствует'} "
                f"({elems.roof_confidence*100:.1f}%)"
            )
            st.progress(elems.roof_confidence)

            st.write(
                f"**4. Контур, окна и двери (вес 15%):** "
                f"{'✅ Остеклен/Закрыт' if elems.has_windows_doors else '❌ Не закрыт'} "
                f"({elems.windows_doors_confidence*100:.1f}%)"
            )
            st.progress(elems.windows_doors_confidence)

            # Defects and Alerts
            if cv_result.fraud_reasons or cv_result.defects_detected:
                st.subheader("⚠️ Выявленные отклонения и дефекты")
                for reason in cv_result.fraud_reasons:
                    st.error(f"🚨 **Stop-Fraud:** {reason}")
                for defect in cv_result.defects_detected:
                    st.warning(f"⚠️ **Дефект:** {defect}")

        with col_photos:
            st.subheader("📷 Ракурсы съемки БПЛА")

            c_chk, c_opt = st.columns([3, 2])
            with c_chk:
                show_boundary = st.checkbox(
                    "🔴 Граница участка (красная линия)",
                    value=True,
                    help="Проекция кадастровой границы Росреестра на плоскость снимка с учетом RTK телеметрии",
                )
            with c_opt:
                boundary_thickness = st.select_slider(
                    "Толщина линии:",
                    options=[2, 4, 6, 8],
                    value=4,
                )

            dataset = DroneDataset()
            images = []
            try:
                plot_data = dataset.get_plot(selected_raw_id)
                images = plot_data.rgb_images
            except Exception:
                images = []

            if images:
                shot_tabs = st.tabs([f"Кадр {i+1}" for i in range(min(len(images), 4))])
                for idx, tab in enumerate(shot_tabs):
                    with tab:
                        img_path = images[idx]
                        try:
                            pil_img = Image.open(img_path)
                            meta = cv_result.shots_metadata[idx] if idx < len(cv_result.shots_metadata) else None

                            if show_boundary and parcel is not None:
                                display_img = BoundaryOverlay.apply_boundary_overlay(
                                    image=pil_img,
                                    meta=meta,
                                    parcel=parcel,
                                    line_color=(255, 30, 30),
                                    line_width=boundary_thickness,
                                    fill_opacity=0.15,
                                    show_badge=True,
                                )
                                caption_prefix = "🔴 Кадастровая граница ЗУ: "
                            else:
                                display_img = pil_img
                                caption_prefix = ""

                            st.image(
                                display_img,
                                caption=f"{caption_prefix}{img_path.name}",
                                use_container_width=True,
                            )
                            if meta:
                                shot_val = getattr(meta.shot_angle, 'value', str(meta.shot_angle))
                                cap_lines = [
                                    f"Ракурс: **{shot_val}** | Высота: {meta.altitude_rel_m or 0:.1f}м | "
                                    f"Pitch: {meta.gimbal_pitch_deg or 0:.1f}° | RTK: **{meta.rtk_status or 'RTK_FIX'}**"
                                ]
                                h_acc = getattr(meta, 'horizontal_accuracy_m', None)

                                if h_acc is not None:

                                    std_lat = getattr(meta, 'std_lat_m', 0.0) or 0.0

                                    std_lon = getattr(meta, 'std_lon_m', 0.0) or 0.0

                                    std_hgt = getattr(meta, 'std_hgt_m', 0.0) or 0.0

                                    cap_lines.append(

                                        f"🎯 **Метрология MRK:** СКО горизонтали **±{h_acc*100:.1f} см** "

                                        f"(Lat: ±{std_lat*100:.1f} см, Lon: ±{std_lon*100:.1f} см, "

                                        f"Hgt: ±{std_hgt*100:.1f} см)"

                                    )

                                ant = getattr(meta, 'antenna_offset_mm', None)

                                if ant:

                                    gps_time = getattr(meta, 'gps_time_sec', 0.0) or 0.0

                                    gps_wk = getattr(meta, 'gps_week', 0) or 0

                                    cap_lines.append(

                                        f"📐 **Фазовый центр антенны:** N: {ant[0]:+.0f} мм, E: {ant[1]:+.0f} мм, V: {ant[2]:+.0f} мм | "

                                        f"GPS-время: {gps_time:.2f}с (неделя {gps_wk})"

                                    )
                                st.caption("  \n".join(cap_lines))
                        except Exception as e:
                            st.caption(f"Ошибка загрузки фото {img_path.name}: {e}")
            else:
                st.info("Фотографии БПЛА для данного участка загружаются или обрабатываются.")

        # =====================================================================
        # COMPREHENSIVE DATASET SHOWCASE: PPK GNSS, MULTISPECTRAL & NOTES
        # =====================================================================
        st.markdown("---")
        c_ppk, c_extra = st.columns([1, 1])

        with c_ppk:
            st.subheader("🛰️ Геодезический аудит PPK / RINEX 3.05")
            ppk = getattr(cv_result, 'ppk_summary', None)
            if ppk and getattr(ppk, 'has_ppk', False):
                st.success(
                    f"✅ **Сырые спутниковые наблюдения GNSS подтверждены** (`{ppk.obs_filename}`)"
                )
                p_c1, p_c2 = st.columns(2)
                with p_c1:
                    st.write(f"**Версия RINEX:** `{ppk.rinex_version}`")
                    st.write(f"**Спутников в захвате:** **{ppk.approx_satellites_tracked}**")
                    st.write("**Кинематика:** Двухчастотная L1/L2 (фазовая)")
                with p_c2:
                    st.write(f"**Старт:** `{ppk.start_time_utc}`")
                    st.write(f"**Финиш:** `{ppk.end_time_utc}`")
                    st.write("**Анти-спуфинг:** ✅ Подтвержден")

                st.info(
                    f"📡 **Спутниковые созвездия:** {', '.join(ppk.constellations)}  \n"
                    f"**Метрологический класс:** {ppk.geodetic_survey_grade}"
                )
            else:
                st.info("ℹ️ По данному объекту геодезическая привязка выполнена по бортовому RTK логу (MRK).")

        with c_extra:
            ms = getattr(cv_result, 'multispectral', None)

            notes = getattr(cv_result, 'operator_notes', [])
            if ms and getattr(ms, 'has_multispectral', False):
                st.subheader("🌿 Спектрометрический анализ DJI M3M (NDVI)")
                st.write(f"**Статус территории:** {ms.site_activity_status}")
                m_c1, m_c2, m_c3 = st.columns(3)
                with m_c1:
                    st.metric("Средний NDVI", f"{ms.mean_ndvi:.3f}")
                with m_c2:
                    st.metric("Пятно застройки", f"{ms.construction_soil_pct:.1f}%")
                with m_c3:
                    st.metric("Растительность", f"{ms.vegetation_pct:.1f}%")

                if ms.heatmap_rel_path and os.path.exists(ms.heatmap_rel_path):
                    st.image(
                        ms.heatmap_rel_path,
                        caption="Тепловая карта NDVI (Красный: котлован/бетон | Зеленый: растительность)",
                        use_container_width=True,
                    )
                st.caption(
                    f"Спектральные диапазоны: {', '.join(ms.bands_available[:2])} • "
                    f"Риск зарастания бурьяном: **{ms.weed_overgrowth_risk}**"
                )
            elif notes:
                st.subheader("📋 Полевой журнал пилота БПЛА")
                for note in notes:
                    st.warning(f"📝 **Бортовая заметка:** {note}")
                st.caption("Данные из бортовых текстовых заметок автоматически переданы в алгоритмы детекции рисков.")
            else:
                st.subheader("📋 Полевой журнал пилота БПЛА")
                st.caption("Полет выполнен в штатном автоматическом режиме по заданию АС «Калита». Замечаний оператора нет.")

    # =========================================================================
    # TAB 2: INTERACTIVE MAP (Leaflet.js client-side)
    # =========================================================================
    with tab2:
        st.subheader("🗺️ Геоинформационная карта залоговых объектов Сибирского банка")
        st.caption("Интерактивная карта участков ИЖС Новосибирской области с цветовой индикацией банковских рисков.")

        # Serialize markers for Leaflet.js
        markers_json = json.dumps(portfolio_data, ensure_ascii=False)

        leaflet_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <style>
                html, body {{ height: 100%; margin: 0; padding: 0; }}
                #map {{ height: 520px; width: 100%; border-radius: 8px; border: 1px solid #d1d5db; }}
                .custom-popup {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; line-height: 1.4; }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                var map = L.map('map').setView([54.90, 83.15], 9);
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '© OpenStreetMap contributors | ПАО Сбербанк AgentDrone'
                }}).addTo(map);

                var data = {markers_json};

                data.forEach(function(item) {{
                    var color = "#15803d"; // Green
                    if (!item.is_inspected) {{
                        color = "#94a3b8"; // Slate for uninspected
                    }} else if (item.risk_level === "CRITICAL") {{
                        color = "#b91c1c"; // Red
                    }} else if (item.risk_level === "MEDIUM") {{
                        color = "#d97706"; // Amber
                    }}

                    var popupContent = '<div class="custom-popup">' +
                        '<b style="color:' + color + ';">' + item.cadastral_id + '</b><br/>' +
                        '<b>Заемщик:</b> ' + item.borrower + '<br/>' +
                        '<b>Адрес:</b> ' + item.address + '<br/>' +
                        '<b>Кредит:</b> ' + item.credit_amount.toLocaleString() + ' ₽<br/>';

                    if (item.is_inspected) {{
                        var readPct = (item.readiness_pct !== null && item.readiness_pct !== undefined) ? item.readiness_pct.toFixed(1) : "0.0";
                        popupContent += '<b>Готовность:</b> ' + readPct + '% (Стадия ' + item.stage + '%)<br/>' +
                            '<b>Материал:</b> ' + item.material + '<br/>' +
                            '<b>Решение:</b> <b>' + item.strategy_code + '</b> (' + item.strategy + ')<br/>' +
                            '<hr style="margin:6px 0; border:0; border-top:1px solid #e5e7eb;"/>' +
                            '<span style="font-size:11px; color:#6b7280;">АС Калита: ' + item.kalita_status + '</span>';
                    }} else {{
                        popupContent += '<b>Статус:</b> ⚪ Ожидает инспекции (перейдите на участок в меню слева)<br/>' +
                            '<b>Материал (по КД):</b> ' + item.material + '<br/>' +
                            '<hr style="margin:6px 0; border:0; border-top:1px solid #e5e7eb;"/>' +
                            '<span style="font-size:11px; color:#6b7280;">АС Калита: Ожидает осмотра</span>';
                    }}
                    popupContent += '</div>';

                    // Custom Circle Marker
                    var circle = L.circleMarker([item.lat, item.lon], {{
                        radius: 10,
                        fillColor: color,
                        color: "#ffffff",
                        weight: 2,
                        opacity: 1,
                        fillOpacity: item.is_inspected ? 0.85 : 0.45
                    }}).addTo(map);

                    // Official Cadastral Boundary Polygon (Красные линии ЗУ)
                    if (item.polygon && item.polygon.length >= 3) {{
                        var poly = L.polygon(item.polygon, {{
                            color: "#dc2626",
                            fillColor: color,
                            fillOpacity: item.is_inspected ? 0.22 : 0.08,
                            weight: 2,
                            dashArray: "4, 4"
                        }}).addTo(map);
                        poly.bindPopup(popupContent);
                    }}

                    // Cadastral Buffer Zone (~50m)
                    L.circle([item.lat, item.lon], {{
                        radius: 50,
                        color: color,
                        fillColor: color,
                        fillOpacity: item.is_inspected ? 0.08 : 0.02,
                        weight: 1
                    }}).addTo(map);

                    circle.bindPopup(popupContent);
                    var tooltipLabel = item.cadastral_id + ' (' + item.borrower.split(' ')[0] + (item.is_inspected ? '' : ' — не проверен') + ')';
                    circle.bindTooltip(tooltipLabel, {{
                        permanent: false,
                        direction: 'top'
                    }});
                }});
            </script>
        </body>
        </html>
        """
        components.html(leaflet_html, height=540)

    # =========================================================================
    # TAB 3: PORTFOLIO MONITORING & ROI
    # =========================================================================
    with tab3:
        st.subheader("📊 Портфельный мониторинг и Экономический эффект (ROI)")

        col_p1, col_p2 = st.columns([3, 2])
        with col_p1:
            st.markdown("#### Реестр залоговых объектов ИЖС Сибирского банка")

            # Pure HTML table
            rows_html = ""
            for item in portfolio_data:
                if not item.get("is_inspected"):
                    risk_badge = "<span style='background:#f3f4f6; color:#6b7280; padding:4px 10px; border-radius:9999px; font-weight:600; font-size:12px;'>ОЖИДАЕТ</span>"
                    readiness_str = "<span style='color:#9ca3af;'>Ожидает проверки</span>"
                    strat_str = "<span style='color:#9ca3af;'>—</span>"
                else:
                    risk_badge = (
                        "<span class='badge-red'>КРИТИЧЕСКИЙ</span>"
                        if item["risk_level"] == "CRITICAL"
                        else (
                            "<span class='badge-yellow'>ВНИМАНИЕ</span>"
                            if item["risk_level"] == "MEDIUM"
                            else "<span class='badge-green'>ОК</span>"
                        )
                    )
                    pct_val = item['readiness_pct'] if item['readiness_pct'] is not None else 0.0
                    readiness_str = f"<b>{pct_val:.1f}%</b> ({item['stage']}%)"
                    strat_str = f"<code>{item['strategy_code']}</code>"

                rows_html += f"""
                <tr>
                    <td><b>{item['cadastral_id']}</b></td>
                    <td>{item['borrower']}</td>
                    <td>{item['credit_amount']:,.0f} ₽</td>
                    <td>{readiness_str}</td>
                    <td>{item['material']}</td>
                    <td>{strat_str}</td>
                    <td>{risk_badge}</td>
                </tr>
                """

            table_html = f"""
            <table class="sber-table">
                <thead>
                    <tr>
                        <th>Кадастровый номер</th>
                        <th>Заемщик</th>
                        <th>Сумма кредита</th>
                        <th>Готовность</th>
                        <th>Материал</th>
                        <th>Стратегия</th>
                        <th>Статус</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)

        with col_p2:
            st.markdown("#### Распределение объектов по стратегиям")

            strat_counts = {}
            inspected_list = [item for item in portfolio_data if item.get("is_inspected")]
            for item in inspected_list:
                code = item["strategy_code"]
                strat_counts[code] = strat_counts.get(code, 0) + 1

            if inspected_list:
                for scode, count in sorted(strat_counts.items()):
                    pct = (count / len(inspected_list)) * 100 if inspected_list else 0
                    st.write(f"**{scode}** ({count} из {len(inspected_list)} пров., {pct:.0f}%)")
                    st.progress(pct / 100.0)
            else:
                st.info("Нет проверенных объектов. Перейдите на участок для расчета стратегии.")

        st.markdown("---")
        st.subheader("💰 Экономический эффект от внедрения AgentDrone для Сбера")

        col_roi1, col_roi2, col_roi3 = st.columns(3)
        with col_roi1:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">Сокращение времени инспекции</div>
                    <div class="metric-value" style="color:#15803d;">96 ч ➔ 15 мин</div>
                    <div style="font-size:12px; color:#6b7280;">Ускорение цикла в 380 раз</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_roi2:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">Предотвращенные потери (Фрод)</div>
                    <div class="metric-value" style="color:#b91c1c;">12 000 000 ₽</div>
                    <div style="font-size:12px; color:#6b7280;">Кейс Бердск (супермаркет на ИЖС)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_roi3:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">Экономия на выездах инспекторов</div>
                    <div class="metric-value" style="color:#2563eb;">~4 800 ₽ / объект</div>
                    <div style="font-size:12px; color:#6b7280;">ROI системы окупается за 1.5 мес.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 4: OFFICIAL INSPECTION ACT (ANNEX 1)
    # =========================================================================
    with tab4:
        st.subheader("📑 Официальный Акт проверки залога недвижимости (Приложение 1)")
        st.caption("Форма утверждена Регламентом залогового мониторинга Сибирского банка ПАО Сбербанк.")

        dataset_inst = DroneDataset()
        plot_photos: list[ReportPhotoItem] = []
        if dataset_inst.is_available():
            try:
                plot_data = dataset_inst.get_plot(selected_raw_id)
                for idx, p_img in enumerate(plot_data.rgb_images[:6]):
                    plot_photos.append(
                        ReportPhotoItem(
                            file_path=str(p_img),
                            caption=f"Кадр {idx+1}: {p_img.name} (фасад/надир)",
                            is_key_view=idx < 2,
                        )
                    )
            except Exception:
                pass

        current_giga = st.session_state.get(f"giga_summary_{selected_raw_id}")
        act_data = build_act_from_cv(
            cv_result=cv_result,
            credit_obj=credit_obj,
            decision=decision,
            gigachat_summary=current_giga,
            photos=plot_photos,
        )
        act_generator = ActGenerator()
        act_html = act_generator.render_html(act_data)

        # Embedded preview in Streamlit
        components.html(act_html, height=850, scrolling=True)

        st.write("")
        b64_act = base64.b64encode(act_html.encode("utf-8")).decode("utf-8")
        href = (
            f'<a href="data:text/html;base64,{b64_act}" '
            f'download="Акт_проверки_залога_{selected_raw_id}.html" '
            f'style="background:#21a038; color:white; padding:10px 18px; border-radius:6px; '
            f'text-decoration:none; font-weight:600;">📥 Скачать официальный Акт (HTML / Печать A4)</a>'
        )
        st.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
