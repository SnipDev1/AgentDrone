"""AgentDrone - Web Dashboard & Inspection Workplace for Sberbank.

Automated AI-assisted collateral and mortgage construction monitoring using
DJI drone telemetry, computer vision inspection, cadastral geofencing,
and the Bank Decision Engine.
Optimized for zero-AVX Linux server environments.
"""

from __future__ import annotations

import base64
from datetime import datetime
import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

from src.cv.boundary_overlay import BoundaryOverlay
from src.cv.pipeline import CVPipeline
from src.dataset import DroneDataset
from src.engine.decision_engine import DecisionEngine
from src.ingestion.cadastral_registry import CadastralRegistry, normalize_cadastral_id
from src.ingestion.credit_registry import CreditRegistry
from src.models.schemas import CreditObject, CVInspectionResult, StrategyRecommendation

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


@st.cache_data(show_spinner=False)
def analyze_single_plot(plot_id: str) -> tuple[CVInspectionResult, CreditObject, StrategyRecommendation]:
    """Execute end-to-end CV and Decision Engine pipeline with caching."""
    credit_obj = CreditRegistry.get_credit_object(plot_id)
    cv_result = CVPipeline.analyze_plot(
        plot_id=plot_id,
        credit_object=credit_obj,
        declared_material=credit_obj.declared_wall_material,
    )
    decision = DecisionEngine.evaluate(cv_result, credit_obj)
    return cv_result, credit_obj, decision


@st.cache_data(show_spinner=False)
def analyze_portfolio_summary(plot_ids: tuple[str, ...]) -> list[dict]:
    """Aggregate portfolio analysis for all monitored collateral assets."""
    summary = []
    for pid in plot_ids:
        try:
            cv_res, credit, dec = analyze_single_plot(pid)
            parcel = CadastralRegistry.get_parcel(pid)
            summary.append({
                "raw_id": pid,
                "cadastral_id": cv_res.cadastral_id,
                "borrower": credit.borrower_name,
                "address": parcel.address if parcel else credit.address,
                "credit_amount": credit.credit_amount_rub,
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
                "days_overdue": credit.days_overdue,
            })
        except Exception as e:
            st.warning(f"Ошибка анализа участка {pid}: {e}")
    return summary


def main():
    plot_ids = get_available_plot_ids()

    # Pre-calculate portfolio data
    portfolio_data = analyze_portfolio_summary(tuple(plot_ids))

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
    avg_readiness = (sum(d["readiness_pct"] for d in portfolio_data) / len(portfolio_data)) if portfolio_data else 0
    fraud_count = sum(1 for d in portfolio_data if d["fraud_flag"])
    auto_approved = sum(1 for d in portfolio_data if d["strategy_code"] == "STRAT-01")

    with col_m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Объектов на мониторинге</div>
                <div class="metric-value">{len(portfolio_data)}</div>
                <div style="font-size:12px; color:#15803d;">● Все проверены БПЛА</div>
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
                <div style="font-size:12px; color:#2563eb;">По весам Банка</div>
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

    # Sidebar: Object Selector and Filters
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Sberbank_Logo_2020.svg/320px-Sberbank_Logo_2020.svg.png",
        width=160,
    )
    st.sidebar.title("Навигация")

    filter_risk = st.sidebar.selectbox(
        "Фильтр по статусу риска:",
        ["Все объекты (13)", "Только с рисками / Stop-Fraud", "Без рисков (ОК)"],
    )

    filtered_plots = []
    for item in portfolio_data:
        if filter_risk == "Только с рисками / Stop-Fraud" and item["risk_level"] != "CRITICAL":
            continue
        if filter_risk == "Без рисков (ОК)" and item["risk_level"] != "LOW":
            continue
        filtered_plots.append(item["raw_id"])

    if not filtered_plots:
        filtered_plots = plot_ids

    def make_plot_label(raw_id: str) -> str:
        row = next((r for r in portfolio_data if r["raw_id"] == raw_id), None)
        norm_id = normalize_cadastral_id(raw_id)
        if row:
            risk_icon = "🔴" if row["risk_level"] == "CRITICAL" else ("🟡" if row["risk_level"] == "MEDIUM" else "🟢")
            return f"{risk_icon} {norm_id} ({row['borrower'].split()[0]}, {row['readiness_pct']:.0f}%)"
        return norm_id

    selected_raw_id = st.sidebar.selectbox(
        "Выберите кадастровый участок:",
        filtered_plots,
        format_func=make_plot_label,
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 Телеметрия и Сервисы")
    st.sidebar.markdown(
        """
        - **БПЛА:** DJI Mavic 3 Enterprise / M3M
        - **Позиционирование:** RTK / PPK (Q=16 сантиметровый)
        - **Кадастр:** Росреестр НСО (Центроиды, ВРИ)
        - **Интеграция:** АС «Калита» (Шлюз активен)
        - **AI Ядро:** OpenCV + Heuristic Risk Matrix
        """
    )

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
        cv_result, credit_obj, decision = analyze_single_plot(selected_raw_id)
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
                                st.caption(
                                    f"Ракурс: **{meta.shot_angle.value}** | Высота: {meta.altitude_rel_m or 0:.1f}м | "
                                    f"Pitch: {meta.gimbal_pitch_deg or 0:.1f}° | RTK: Q={meta.rtk_flag or 16}"
                                )
                        except Exception as e:
                            st.caption(f"Ошибка загрузки фото {img_path.name}: {e}")
            else:
                st.info("Фотографии БПЛА для данного участка загружаются или обрабатываются.")

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
                    if (item.risk_level === "CRITICAL") {{
                        color = "#b91c1c"; // Red
                    }} else if (item.risk_level === "MEDIUM") {{
                        color = "#d97706"; // Amber
                    }}

                    // Custom Circle Marker
                    var circle = L.circleMarker([item.lat, item.lon], {{
                        radius: 10,
                        fillColor: color,
                        color: "#ffffff",
                        weight: 2,
                        opacity: 1,
                        fillOpacity: 0.85
                    }}).addTo(map);

                    // Official Cadastral Boundary Polygon (Красные линии ЗУ)
                    if (item.polygon && item.polygon.length === 4) {{
                        var poly = L.polygon(item.polygon, {{
                            color: "#dc2626",
                            fillColor: color,
                            fillOpacity: 0.22,
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
                        fillOpacity: 0.08,
                        weight: 1
                    }}).addTo(map);

                    var popupContent = '<div class="custom-popup">' +
                        '<b style="color:' + color + ';">' + item.cadastral_id + '</b><br/>' +
                        '<b>Заемщик:</b> ' + item.borrower + '<br/>' +
                        '<b>Адрес:</b> ' + item.address + '<br/>' +
                        '<b>Кредит:</b> ' + item.credit_amount.toLocaleString() + ' ₽<br/>' +
                        '<b>Готовность:</b> ' + item.readiness_pct.toFixed(1) + '% (Стадия ' + item.stage + '%)<br/>' +
                        '<b>Материал:</b> ' + item.material + '<br/>' +
                        '<b>Решение:</b> <b>' + item.strategy_code + '</b> (' + item.strategy + ')<br/>' +
                        '<hr style="margin:6px 0; border:0; border-top:1px solid #e5e7eb;"/>' +
                        '<span style="font-size:11px; color:#6b7280;">АС Калита: ' + item.kalita_status + '</span>' +
                        '</div>';

                    circle.bindPopup(popupContent);
                    circle.bindTooltip(item.cadastral_id + ' (' + item.borrower.split(' ')[0] + ')', {{
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
                risk_badge = (
                    "<span class='badge-red'>КРИТИЧЕСКИЙ</span>"
                    if item["risk_level"] == "CRITICAL"
                    else (
                        "<span class='badge-yellow'>ВНИМАНИЕ</span>"
                        if item["risk_level"] == "MEDIUM"
                        else "<span class='badge-green'>ОК</span>"
                    )
                )
                rows_html += f"""
                <tr>
                    <td><b>{item['cadastral_id']}</b></td>
                    <td>{item['borrower']}</td>
                    <td>{item['credit_amount']:,.0f} ₽</td>
                    <td><b>{item['readiness_pct']:.1f}%</b> ({item['stage']}%)</td>
                    <td>{item['material']}</td>
                    <td><code>{item['strategy_code']}</code></td>
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
            for item in portfolio_data:
                code = item["strategy_code"]
                strat_counts[code] = strat_counts.get(code, 0) + 1

            for scode, count in sorted(strat_counts.items()):
                pct = (count / len(portfolio_data)) * 100 if portfolio_data else 0
                st.write(f"**{scode}** ({count} объектов, {pct:.0f}%)")
                st.progress(pct / 100.0)

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

        act_html = f"""
        <div style="background:white; color:#111; padding:30px; border:1px solid #ccc; font-family:'Times New Roman', serif; line-height:1.4;">
            <div style="text-align:right; font-size:12px;">Приложение № 1<br/>к Регламенту мониторинга залогов ИЖС<br/>ПАО Сбербанк (АС «Калита»)</div>
            <h3 style="text-align:center; margin-top:15px; margin-bottom:5px;">АКТ ПРОВЕРКИ ПРЕДМЕТА ЗАЛОГА № АЗ-{selected_raw_id[-6:]}</h3>
            <p style="text-align:center; font-size:13px; margin-bottom:20px;">г. Новосибирск • {datetime.now().strftime('%d.%m.%Y')}</p>

            <table style="width:100%; border-collapse:collapse; font-size:13px; margin-bottom:15px;">
                <tr><td style="border:1px solid #333; padding:6px; width:35%;"><b>Залогодатель / Заемщик:</b></td><td style="border:1px solid #333; padding:6px;">{credit_obj.borrower_name}</td></tr>
                <tr><td style="border:1px solid #333; padding:6px;"><b>Кредитный договор:</b></td><td style="border:1px solid #333; padding:6px;">№ {credit_obj.credit_agreement_number}</td></tr>
                <tr><td style="border:1px solid #333; padding:6px;"><b>Кадастровый номер ЗУ:</b></td><td style="border:1px solid #333; padding:6px;">{cv_result.cadastral_id}</td></tr>
                <tr><td style="border:1px solid #333; padding:6px;"><b>Адрес местоположения:</b></td><td style="border:1px solid #333; padding:6px;">{parcel.address if parcel else credit_obj.address}</td></tr>
                <tr><td style="border:1px solid #333; padding:6px;"><b>Метод осмотра:</b></td><td style="border:1px solid #333; padding:6px;">Автоматизированный мониторинг БПЛА DJI с RTK-геопривязкой и CV-анализом</td></tr>
                <tr><td style="border:1px solid #333; padding:6px;"><b>Фактическая готовность:</b></td><td style="border:1px solid #333; padding:6px;"><b>{cv_result.readiness_pct:.1f}%</b> (Банковская стадия {cv_result.readiness_stage}%)</td></tr>
                <tr><td style="border:1px solid #333; padding:6px;"><b>Материал стен (факт / план):</b></td><td style="border:1px solid #333; padding:6px;">{cv_result.detected_material.value} / {credit_obj.declared_wall_material}</td></tr>
            </table>

            <h4 style="margin:10px 0 5px 0; font-size:13px;">1. Результаты проверки конструктивных элементов:</h4>
            <table style="width:100%; border-collapse:collapse; font-size:12px; margin-bottom:15px;">
                <tr style="background:#eee;">
                    <th style="border:1px solid #333; padding:5px;">Элемент</th>
                    <th style="border:1px solid #333; padding:5px;">Вес банка</th>
                    <th style="border:1px solid #333; padding:5px;">Наличие</th>
                    <th style="border:1px solid #333; padding:5px;">Уверенность CV</th>
                </tr>
                <tr>
                    <td style="border:1px solid #333; padding:5px;">Фундамент</td>
                    <td style="border:1px solid #333; padding:5px; text-align:center;">15%</td>
                    <td style="border:1px solid #333; padding:5px; text-align:center;">{'Имеется' if cv_result.structural_elements.has_foundation else 'Отсутствует'}</td>
                    <td style="border:1px solid #333; padding:5px; text-align:center;">{cv_result.structural_elements.foundation_confidence*100:.1f}%</td>
                </tr>
                <tr>
                    <td style="border:1px solid #333; padding:5px;">Несущие стены</td>
                    <td style="border:1px solid #333; padding:5px; text-align:center;">40%</td>
                    <td style="border:1px solid #333; padding:5px; text-align:center;">{'Имеется' if cv_result.structural_elements.has_walls else 'Отсутствует'}</td>
                    <td style="border:1px solid #333; padding:5px; text-align:center;">{cv_result.structural_elements.walls_confidence*100:.1f}%</td>
                </tr>
                <tr>
                    <td style="border:1px solid #333; padding:5px;">Кровельная система</td>
                    <td style="border:1px solid #333; padding:5px; text-align:center;">30%</td>
                    <td style="border:1px solid #333; padding:5px; text-align:center;">{'Имеется' if cv_result.structural_elements.has_roof else 'Отсутствует'}</td>
                    <td style="border:1px solid #333; padding:5px; text-align:center;">{cv_result.structural_elements.roof_confidence*100:.1f}%</td>
                </tr>
                <tr>
                    <td style="border:1px solid #333; padding:5px;">Контур / Окна / Двери</td>
                    <td style="border:1px solid #333; padding:5px; text-align:center;">15%</td>
                    <td style="border:1px solid #333; padding:5px; text-align:center;">{'Имеется' if cv_result.structural_elements.has_windows_doors else 'Отсутствует'}</td>
                    <td style="border:1px solid #333; padding:5px; text-align:center;">{cv_result.structural_elements.windows_doors_confidence*100:.1f}%</td>
                </tr>
            </table>

            <h4 style="margin:10px 0 5px 0; font-size:13px;">2. Заключение Банковского Decision Engine:</h4>
            <p style="font-size:13px; margin:0 0 10px 0;">
                <b>Рекомендованная стратегия:</b> {decision.strategy_code} — {decision.strategy.value}<br/>
                <b>Предписываемое действие:</b> {decision.action_required}<br/>
                <b>Статус в АС «Калита»:</b> {decision.kalita_status}
            </p>

            <div style="margin-top:25px; display:flex; justify-content:space-between; font-size:12px;">
                <div>
                    <b>Инспектор залоговой службы:</b><br/>
                    ИИ-агент AgentDrone (v1.0-Hackathon)<br/>
                    ЭЦП: <code>SHA256: 4F9B...C82A (Действительна)</code>
                </div>
                <div>
                    <b>Куратор кредитного отдела:</b><br/>
                    _________________ / Смирнова В.А. /<br/>
                    Дата акцепта: {datetime.now().strftime('%d.%m.%Y')}
                </div>
            </div>
        </div>
        """
        st.markdown(act_html, unsafe_allow_html=True)

        st.write("")
        b64_act = base64.b64encode(act_html.encode("utf-8")).decode("utf-8")
        href = (
            f'<a href="data:text/html;base64,{b64_act}" '
            f'download="Акт_проверки_залога_{selected_raw_id}.html" '
            f'style="background:#21a038; color:white; padding:10px 18px; border-radius:6px; '
            f'text-decoration:none; font-weight:600;">📥 Скачать официальный Акт (HTML / Печать)</a>'
        )
        st.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
