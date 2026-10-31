"""AgentDrone - Web Dashboard & Inspection Workplace for Sberbank.

Automated AI-assisted collateral and mortgage construction monitoring using
DJI drone telemetry, computer vision inspection, cadastral geofencing,
and the Bank Decision Engine.
Optimized for zero-AVX Linux server environments with Sberbank Plasma Design System.
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
from src.ui.sber_theme import (
    SBER_LOGO_SVG,
    get_sber_css,
    render_gigachat_card,
    render_kpi_card,
    render_sber_header,
)

# Page configuration
st.set_page_config(
    page_title="AgentDrone — Мониторинг залогов БПЛА | Сбер",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Sberbank Design System (Plasma / SB Sans / Corporate Styling)
st.markdown(get_sber_css(), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_available_plot_ids() -> list[str]:
    """Retrieve all available cadastral plot IDs."""
    dataset = DroneDataset()
    plots = dataset.list_plots()
    if not plots:
        return [p.raw_folder_name for p in CadastralRegistry.list_all_parcels()]
    return plots


CACHE_SALT = "v1.2-sber-multispec"


@st.cache_data(show_spinner=False)
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

    # GigaChat service initialization
    giga_service = GigaChatInspectorService()
    giga_status = giga_service.get_mode_status()

    # Sidebar: Logo & Navigation
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #E5E8ED; margin-bottom: 16px;">
                {SBER_LOGO_SVG}
                <div style="font-size: 13px; color: #687182; margin-top: 6px; font-weight: 500;">
                    Залоговый мониторинг БПЛА
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Навигация по залогам")

        filter_risk = st.selectbox(
            "Фильтр по статусу риска:",
            ["Все объекты (13)", "Только проверенные", "Ожидают проверки", "Только с рисками / Stop-Fraud", "Без рисков (ОК)"],
        )

    # Current portfolio data assembly
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

    with st.sidebar:
        selected_raw_id = st.selectbox(
            "Выберите кадастровый участок:",
            filtered_plots,
            format_func=make_plot_label,
            index=0,
        )

    # Navigation trigger: update plot ONLY when the user navigates to it ("переходит")
    is_new_navigation = (st.session_state.last_visited_plot_id != selected_raw_id)
    is_not_inspected = (selected_raw_id not in st.session_state.inspected_plots)

    if is_new_navigation or is_not_inspected:
        with st.spinner(f"🛸 Обновление данных БПЛА по участку {normalize_cadastral_id(selected_raw_id)}..."):
            cv_result, credit_obj, decision = analyze_single_plot(selected_raw_id)
            st.session_state.inspected_plots[selected_raw_id] = (cv_result, credit_obj, decision)
            st.session_state.last_visited_plot_id = selected_raw_id
        # Re-build portfolio summary with freshly inspected data and rerun to update dropdown badges
        portfolio_data = get_portfolio_summary(plot_ids, st.session_state.inspected_plots)
        st.rerun()
    else:
        cv_result, credit_obj, decision = st.session_state.inspected_plots[selected_raw_id]

    # Sberbank Top Navbar with GigaChat & RTK Status
    st.markdown(
        render_sber_header(
            active_objects_count=len(portfolio_data),
            rtk_status="FIX Q=16",
            giga_status=giga_status["model"] if giga_status["is_online"] else "GigaChat AI (Offline)",
            giga_online=giga_status["is_online"],
        ),
        unsafe_allow_html=True,
    )

    # Top KPI Metrics bar (SberBank Online Widgets)
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
            render_kpi_card(
                title="Объектов проверено",
                value=f"{inspected_count} / {len(portfolio_data)}",
                subtext="● Навигация по переходу",
                icon_type="drone",
                is_success=True,
            ),
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            render_kpi_card(
                title="Кредитный портфель",
                value=f"{total_portfolio_rub / 1_000_000:.1f} млн ₽",
                subtext="100% залоговое покрытие",
                icon_type="money",
            ),
            unsafe_allow_html=True,
        )
    with col_m3:
        st.markdown(
            render_kpi_card(
                title="Средняя готовность",
                value=f"{avg_readiness:.1f}%",
                subtext=f"По проверенным ({inspected_count})",
                icon_type="chart",
            ),
            unsafe_allow_html=True,
        )
    with col_m4:
        st.markdown(
            render_kpi_card(
                title="Алерты Stop-Fraud",
                value=str(fraud_count),
                subtext="⚠️ Требуется блокировка" if fraud_count > 0 else "● Рисков фрода нет",
                icon_type="shield_alert",
                is_alert=fraud_count > 0,
                is_success=fraud_count == 0,
            ),
            unsafe_allow_html=True,
        )
    with col_m5:
        st.markdown(
            render_kpi_card(
                title="Авто-выдача траншей",
                value=str(auto_approved),
                subtext="Прямой акцепт без выезда",
                icon_type="check_circle",
                is_success=True,
            ),
            unsafe_allow_html=True,
        )

    st.write("")

    # Sidebar: Services, Equipment & GigaChat
    with st.sidebar:
        st.markdown("---")
        st.markdown("#### 📡 Сервисы и Оборудование")
        st.markdown(
            """
            <div style="font-size:13px; line-height:1.6; color:#4B5563; background:#F8FAFC; padding:12px; border-radius:12px; border:1px solid #E2E8F0;">
                <div>🛸 <b>БПЛА:</b> DJI Mavic 3 Enterprise / M3M</div>
                <div>🛰️ <b>RTK/PPK:</b> Сантиметровый фикс (Q=16)</div>
                <div>🗺️ <b>Росреестр:</b> Центроиды и ВРИ НСО</div>
                <div>⚡ <b>АС «Калита»:</b> Авто-акцепт траншей</div>
                <div>🧠 <b>AI:</b> Computer Vision + Sber Rules</div>
                <div>🤖 <b>LLM:</b> GigaChat-2-Pro (Сбер)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("#### 🤖 Sber GigaChat API")
        if giga_status["is_online"]:
            st.markdown(
                f"""
                <div class="sber-badge-green" style="margin-bottom:8px;">
                    🟢 {giga_status['mode']} ({giga_status['model']})
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="sber-badge-yellow" style="margin-bottom:8px;">
                    🟡 Автономный шаблонизатор
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("⚙️ Настройки GigaChat API"):
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

        st.markdown("---")
        st.markdown("#### ⚡ Управление кэшем")
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("🔄 Обновить", use_container_width=True, help="Принудительно пересчитать выбранный участок"):
                analyze_single_plot.clear(selected_raw_id)
                cv_result, credit_obj, decision = analyze_single_plot(selected_raw_id)
                st.session_state.inspected_plots[selected_raw_id] = (cv_result, credit_obj, decision)
                giga_key = f"giga_summary_{selected_raw_id}"
                if giga_key in st.session_state:
                    del st.session_state[giga_key]
                st.toast(f"Участок {normalize_cadastral_id(selected_raw_id)} обновлен!", icon="✅")
                st.rerun()
        with c_btn2:
            if st.button("🗑️ Сбросить", use_container_width=True, help="Очистить весь кэш проверок"):
                st.cache_data.clear()
                st.session_state.inspected_plots = {}
                st.session_state.last_visited_plot_id = None
                st.rerun()

    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🛸 Инспекция БПЛА & CV Анализ",
        "🗺️ ГИС-карта залогов НСО",
        "📊 Портфельный мониторинг & ROI",
        "📑 Акт проверки залога (Приложение 1)",
        "🌿 Мультиспектральный анализ (DJI M3M)",
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
                    "<div style='text-align:right;'><span class='sber-badge-red'>⚠️ КРИТИЧЕСКИЙ РИСК / СБ</span></div>",
                    unsafe_allow_html=True,
                )
            elif decision.risk_level == "MEDIUM":
                st.markdown(
                    "<div style='text-align:right;'><span class='sber-badge-yellow'>🟡 ТРЕБУЕТ ВНИМАНИЯ</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='text-align:right;'><span class='sber-badge-green'>🟢 БЕЗ РИСКА / СОГЛАСОВАНО</span></div>",
                    unsafe_allow_html=True,
                )

        st.write("")

        # 4 Inspection Metric Cards (Sber Online Style)
        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
        with col_i1:
            st.markdown(
                render_kpi_card(
                    title="Готовность объекта",
                    value=f"{cv_result.readiness_pct:.1f}%",
                    subtext=cv_result.stage_title,
                    icon_type="chart",
                    is_success=cv_result.readiness_pct >= 50,
                ),
                unsafe_allow_html=True,
            )
            st.progress(min(max(cv_result.readiness_pct / 100.0, 0.0), 1.0))
        with col_i2:
            geo_status = "ПОДТВЕРЖДЕНО" if cv_result.geocheck.is_correct_plot else "НАРУШЕНИЕ"
            st.markdown(
                render_kpi_card(
                    title="Геовалидация (RTK)",
                    value=geo_status,
                    subtext=f"Смещение: {cv_result.geocheck.distance_to_cadastre_m:.1f} м (допуск 75м)",
                    icon_type="drone",
                    is_success=cv_result.geocheck.is_correct_plot,
                    is_alert=not cv_result.geocheck.is_correct_plot,
                ),
                unsafe_allow_html=True,
            )
        with col_i3:
            mat_name = cv_result.detected_material.value
            declared_mat = credit_obj.declared_wall_material
            mat_ok = mat_name in declared_mat or "жилой" in declared_mat.lower()
            st.markdown(
                render_kpi_card(
                    title="Материал стен (CV)",
                    value=mat_name,
                    subtext=f"В договоре: {declared_mat}",
                    icon_type="money",
                    is_success=mat_ok,
                    is_alert=not mat_ok,
                ),
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
                render_kpi_card(
                    title="Ракурсы съемки БПЛА",
                    value=comp_badge,
                    subtext=f"Всего кадров БПЛА: {cv_result.analyzed_images_count}",
                    icon_type="satellite",
                    is_success=comp.is_fully_compliant,
                ),
                unsafe_allow_html=True,
            )

        st.write("")

        # DECISION ENGINE CARD (Sber Corporate Box)
        decision_box_class = (
            "danger"
            if decision.risk_level == "CRITICAL"
            else ("warning" if decision.risk_level == "MEDIUM" else "success")
        )
        st.markdown(
            f"""
            <div class="sber-decision-box {decision_box_class}">
                <div class="sber-decision-header">
                    <h3 class="sber-decision-title">
                        <span>🎯 Решение Банка: <b>{decision.strategy_code}</b> — {decision.strategy.value}</span>
                    </h3>
                    <span class="sber-pill-status active">
                        Статус АС «Калита»: <b>{decision.kalita_status}</b>
                    </span>
                </div>
                <p style="margin: 8px 0 10px 0; font-size:15px; font-weight:600; color:#121922;">
                    {decision.action_required}
                </p>
                <div style="font-size:13.5px; color:#4B5563; margin-top:8px;">
                    <strong>Обоснование алгоритма:</strong>
                    <ul style="margin:6px 0 0 18px; padding:0; line-height:1.5;">
                        {''.join(f'<li>{r}</li>' for r in decision.reasoning)}
                    </ul>
                </div>
                <div style="margin-top:12px; padding-top:10px; border-top:1px solid #E5E8ED; font-size:12.5px; color:#687182; display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;">
                    <div><strong>Юридическое основание:</strong> {decision.legal_basis}</div>
                    <div><strong>Финансовый эффект:</strong> <span style="color:#0E7425; font-weight:700;">{decision.financial_impact_rub:,.0f} ₽</span></div>
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

        # GigaChat AI Inspector Summary Card (Sber Theme)
        st.write("")
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

        st.markdown(
            render_gigachat_card(
                summary_text=current_summary,
                is_online=status_info["is_online"],
                model_name=status_info["model"],
            ),
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
                f"{'✅ Закрыт' if elems.has_windows_doors else '❌ Открыт'} "
                f"({elems.windows_doors_confidence*100:.1f}%)"
            )
            st.progress(elems.windows_doors_confidence)

            st.markdown("---")
            st.subheader("🛡️ Детекция рисков (Stop-Fraud)")
            if cv_result.fraud_risk_flag:
                for reason in cv_result.fraud_reasons:
                    st.error(f"⚠️ {reason}")
            else:
                st.success("✅ Признаков нецелевого использования или подмены материалов не обнаружено.")

            if cv_result.defects_detected:
                st.warning("⚠️ Обнаруженные дефекты конструктива:")
                for d in cv_result.defects_detected:
                    st.write(f"- {d}")

        with col_photos:
            st.subheader("📷 Ракурсы съемки БПЛА и Метаданные")
            dataset = DroneDataset()
            if dataset.is_available():
                try:
                    plot_data = dataset.get_plot(selected_raw_id)
                    images = plot_data.rgb_images
                    if images:
                        img_tabs = st.tabs([f"Кадр {i+1}" for i in range(min(len(images), 6))])
                        for i, tab in enumerate(img_tabs):
                            with tab:
                                img_path = images[i]
                                show_3d_boundary = st.checkbox(
                                    "📐 3D Границы кадастра и отступы 3м",
                                    value=True,
                                    key=f"bnd_chk_{selected_raw_id}_{i}",
                                )

                                overlay_img = None
                                if show_3d_boundary:
                                    try:
                                        raw_pil = Image.open(img_path)
                                        overlay_img = BoundaryOverlay.overlay_boundaries(
                                            image=raw_pil,
                                            parcel=parcel,
                                            shot_meta=cv_result.shots_metadata[i] if i < len(cv_result.shots_metadata) else None,
                                        )
                                    except Exception:
                                        overlay_img = None

                                if overlay_img is not None:
                                    st.image(
                                        overlay_img,
                                        caption=f"Кадр {i+1}: {img_path.name} (Зеленый = граница участка, Оранжевый = отступ 3м)",
                                        use_container_width=True,
                                    )
                                else:
                                    st.image(str(img_path), caption=f"Кадр {i+1}: {img_path.name}", use_container_width=True)

                                if i < len(cv_result.shots_metadata):
                                    m = cv_result.shots_metadata[i]
                                    st.caption(
                                        f"Ракурс: **{m.shot_angle.value}** | Pitch: `{m.gimbal_pitch_deg:.1f}°` | "
                                        f"Yaw: `{m.gimbal_yaw_deg:.1f}°` | Высота: `{m.altitude_rel_m:.1f} м` | RTK: `Q={m.rtk_flag}`"
                                    )
                    else:
                        st.info("Нет доступных снимков для отображения.")
                except Exception as e:
                    st.error(f"Ошибка загрузки изображений: {e}")
            else:
                st.info("Датасет БПЛА не смонтирован. Используются эталонные метаданные.")

    # =========================================================================
    # TAB 2: GIS LEAFLET MAP
    # =========================================================================
    with tab2:
        st.subheader("🗺️ Геоинформационная карта залоговых объектов НСО")
        st.caption("Координаты определены с субметровой точностью на основе бортовой RTK-телеметрии БПЛА DJI.")

        map_center_lat = 54.85
        map_center_lon = 83.05

        features_js = []
        for item in portfolio_data:
            if item.get("lat") and item.get("lon"):
                is_selected = (item["raw_id"] == selected_raw_id)
                features_js.append({
                    "id": item["raw_id"],
                    "cadastral_id": item["cadastral_id"],
                    "borrower": item["borrower"],
                    "address": item["address"],
                    "credit_amount": item["credit_amount"],
                    "readiness_pct": item.get("readiness_pct") or 0.0,
                    "stage": item.get("stage", "—"),
                    "material": item.get("material", "—"),
                    "strategy": item.get("strategy", "—"),
                    "strategy_code": item.get("strategy_code", "ОЖИДАЕТ"),
                    "risk_level": item.get("risk_level", "PENDING"),
                    "kalita_status": item.get("kalita_status", "Ожидает"),
                    "lat": item["lat"],
                    "lon": item["lon"],
                    "polygon": item.get("polygon", []),
                    "is_selected": is_selected,
                    "is_inspected": item.get("is_inspected", False),
                })

        leaflet_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <style>
                #map {{ height: 500px; width: 100%; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.06); }}
                .leaflet-popup-content-wrapper {{ border-radius: 12px; font-family: 'SB Sans Text', sans-serif; }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                var map = L.map('map').setView([{map_center_lat}, {map_center_lon}], 10);
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    maxZoom: 18,
                    attribution: '© OpenStreetMap | Сибирский банк ПАО Сбербанк'
                }}).addTo(map);

                var items = {json.dumps(features_js, ensure_ascii=False)};
                items.forEach(function(item) {{
                    var color = '#15803d';
                    if (!item.is_inspected) color = '#9ca3af';
                    else if (item.risk_level === 'CRITICAL') color = '#b91c1c';
                    else if (item.risk_level === 'MEDIUM') color = '#eab308';

                    var radius = item.is_selected ? 14 : 9;

                    if (item.polygon && item.polygon.length > 2) {{
                        L.polygon(item.polygon, {{
                            color: color,
                            fillColor: color,
                            fillOpacity: 0.25,
                            weight: item.is_selected ? 3 : 1.5
                        }}).addTo(map);
                    }}

                    var circle = L.circleMarker([item.lat, item.lon], {{
                        radius: radius,
                        fillColor: color,
                        color: item.is_selected ? '#ffffff' : '#ffffff',
                        weight: item.is_selected ? 3 : 1.5,
                        opacity: 1,
                        fillOpacity: 0.9
                    }}).addTo(map);

                    var popupContent = '<div style="font-size:13px; line-height:1.4;">' +
                        '<strong style="color:' + color + ';">' + item.cadastral_id + '</strong><br/>' +
                        '<b>Заемщик:</b> ' + item.borrower + '<br/>' +
                        '<b>Адрес:</b> ' + item.address + '<br/>' +
                        '<b>Кредит:</b> ' + item.credit_amount.toLocaleString() + ' ₽<br/>' +
                        '<b>Готовность:</b> ' + (item.readiness_pct ? item.readiness_pct.toFixed(1) + '%' : 'Ожидает') + '<br/>' +
                        '<b>Решение:</b> <b>' + item.strategy_code + '</b><br/>' +
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

            # Pure HTML table in Sber style
            rows_html = ""
            for item in portfolio_data:
                if not item.get("is_inspected"):
                    risk_badge = "<span style='background:#f3f4f6; color:#6b7280; padding:4px 10px; border-radius:9999px; font-weight:600; font-size:12px;'>ОЖИДАЕТ</span>"
                    ready_str = "Ожидает"
                else:
                    risk_badge = (
                        "<span class='sber-badge-red'>КРИТИЧЕСКИЙ</span>"
                        if item["risk_level"] == "CRITICAL"
                        else (
                            "<span class='sber-badge-yellow'>ВНИМАНИЕ</span>"
                            if item["risk_level"] == "MEDIUM"
                            else "<span class='sber-badge-green'>ОК</span>"
                        )
                    )
                    ready_str = f"<b>{item['readiness_pct']:.1f}%</b> ({item['stage']}%)"

                rows_html += f"""
                <tr>
                    <td><b>{item['cadastral_id']}</b></td>
                    <td>{item['borrower']}</td>
                    <td>{item['credit_amount']:,.0f} ₽</td>
                    <td>{ready_str}</td>
                    <td>{item['material']}</td>
                    <td><code>{item['strategy_code']}</code></td>
                    <td>{risk_badge}</td>
                </tr>
                """

            table_html = f"""
            <table class="sber-table-v2">
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
                render_kpi_card(
                    title="Сокращение времени инспекции",
                    value="96 ч ➔ 15 мин",
                    subtext="Ускорение мониторинга в 380 раз",
                    icon_type="drone",
                    is_success=True,
                ),
                unsafe_allow_html=True,
            )
        with col_roi2:
            st.markdown(
                render_kpi_card(
                    title="Предотвращенные потери (Фрод)",
                    value="12 000 000 ₽",
                    subtext="Кейс Бердск: супермаркет на ИЖС",
                    icon_type="shield_alert",
                    is_alert=True,
                ),
                unsafe_allow_html=True,
            )
        with col_roi3:
            st.markdown(
                render_kpi_card(
                    title="Экономия на выездах инспектора",
                    value="~4 800 ₽ / объект",
                    subtext="Окупаемость внедрения за 1.5 мес.",
                    icon_type="money",
                    is_success=True,
                ),
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
            f'style="display:inline-flex; align-items:center; gap:8px; '
            f'background:linear-gradient(108deg, #0e8528 0%, #21a038 52%, #18d24e 100%); '
            f'color:white; padding:12px 26px; border-radius:9999px; text-decoration:none; '
            f'font-family:\'SB Sans Text\', sans-serif; font-weight:600; font-size:14px; '
            f'box-shadow: 0 4px 14px rgba(33, 160, 56, 0.3); transition: transform 0.2s ease;">'
            f'📥 Скачать официальный Акт проверки залога (HTML / Печать A4)</a>'
        )
        st.markdown(href, unsafe_allow_html=True)

    # =========================================================================
    # TAB 5: MULTISPECTRAL ANALYSIS (DJI M3M)
    # =========================================================================
    with tab5:
        st.subheader("🌿 Мультиспектральный анализ и Зонирование (DJI Mavic 3 Multispectral)")
        st.caption("Анализ вегетационных индексов (NDVI / NDRE) и детекция признаков заброшенности/долгостроя.")

        multi = cv_result.multispectral
        if multi and multi.is_available:
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.markdown(
                    render_kpi_card(
                        title="Средний индекс NDVI",
                        value=f"{multi.mean_ndvi:.2f}",
                        subtext="Здоровая растительность" if multi.mean_ndvi > 0.4 else "Умеренная растительность",
                        icon_type="chart",
                        is_success=multi.mean_ndvi > 0.4,
                    ),
                    unsafe_allow_html=True,
                )
            with col_m2:
                st.markdown(
                    render_kpi_card(
                        title="Зона застройки",
                        value=f"{multi.built_area_pct:.1f}%",
                        subtext="Пятно строительства",
                        icon_type="drone",
                    ),
                    unsafe_allow_html=True,
                )
            with col_m3:
                st.markdown(
                    render_kpi_card(
                        title="Статус сорняков / бурьяна",
                        value="ОБНАРУЖЕНО" if multi.has_overgrown_vegetation else "В НОРМЕ",
                        subtext="Признак простоя стройки" if multi.has_overgrown_vegetation else "Территория ухожена",
                        icon_type="shield_alert",
                        is_alert=multi.has_overgrown_vegetation,
                        is_success=not multi.has_overgrown_vegetation,
                    ),
                    unsafe_allow_html=True,
                )

            st.write("")
            st.markdown(f"**Агроэкологическое заключение:** {multi.summary_note}")
        else:
            st.info("ℹ️ Для данного участка мультиспектральные каналы (NDRE/NIR) отсутствуют или съемка выполнялась в стандартном RGB-режиме.")


if __name__ == "__main__":
    main()
