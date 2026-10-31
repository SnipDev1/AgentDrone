"""AgentDrone — Сбер Дизайн-система (Plasma / Sberbank UI Theme).

Полное визуальное соответствие официальному порталу sberbank.ru/ru/person:
- Фирменные корпоративные шрифты SB Sans Text & SB Sans Display с официального CDN.
- Цветовая палитра: Sber Green (#21A038), градиенты, светлый фон (#F4F6F9), карточки 20px.
- Векторные SVG-иконки, фирменная шапка с логотипом Сбера, виджеты KPI в стиле СберБанк Онлайн.
- Кастомизация всех нативных компонентов Streamlit под единый корпоративный стиль.
"""

from __future__ import annotations

# Официальный векторный логотип Сбербанка (SVG)
SBER_LOGO_SVG = """
<svg width="140" height="38" viewBox="0 0 160 44" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="sberGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0E8528" />
      <stop offset="55%" stop-color="#21A038" />
      <stop offset="100%" stop-color="#18D24E" />
    </linearGradient>
    <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00A2F8" />
      <stop offset="40%" stop-color="#21A038" />
      <stop offset="85%" stop-color="#F2DD10" />
    </linearGradient>
  </defs>
  <!-- Sber Ring Emblem -->
  <g transform="translate(4, 2)">
    <circle cx="20" cy="20" r="18" stroke="url(#ringGrad)" stroke-width="4" fill="none" stroke-dasharray="100" stroke-dashoffset="15" stroke-linecap="round"/>
    <path d="M12 20 L18 26 L29 14" stroke="url(#sberGrad)" stroke-width="4.2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
  </g>
  <!-- Sber Wordmark -->
  <text x="54" y="29" fill="#121922" font-family="'SB Sans Display', 'SBSansDisplay', sans-serif" font-size="24" font-weight="800" letter-spacing="0.5">СБЕР</text>
</svg>
"""

# Малые SVG иконки в корпоративном стиле
SVG_ICONS = {
    "drone": """
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#21A038" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2a4 4 0 0 0-4 4c0 1.6.9 3 2.3 3.7L7 13.5l1.5 1.5 3.5-3.3 3.5 3.3 1.5-1.5-3.3-3.8A4.1 4.1 0 0 0 16 6a4 4 0 0 0-4-4z"/>
            <path d="m3 7 4 4"/>
            <path d="m21 7-4 4"/>
            <path d="M6 18h12"/>
            <circle cx="12" cy="18" r="2"/>
        </svg>
    """,
    "money": """
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#0098F8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="2" y="6" width="20" height="12" rx="3"/>
            <circle cx="12" cy="12" r="3"/>
            <path d="M6 12h.01M18 12h.01"/>
        </svg>
    """,
    "chart": """
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#21A038" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 3v18h18"/>
            <path d="m7 14 4-4 4 4 5-6"/>
        </svg>
    """,
    "shield_alert": """
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#E11D48" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
    """,
    "check_circle": """
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#16A34A" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
            <polyline points="22 4 12 14.01 9 11.01"/>
        </svg>
    """,
    "satellite": """
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#21A038" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="2"/>
            <path d="M16.24 7.76a6 6 0 0 1 0 8.49m-8.48-.01a6 6 0 0 1 0-8.49"/>
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 19.07a10 10 0 0 1 0-14.14"/>
        </svg>
    """,
}


def get_sber_css() -> str:
    """Возвращает глобальный CSS в стиле sberbank.ru/ru/person."""
    return """
    <link rel="stylesheet" href="https://cdn-app.sberdevices.ru/shared-static/0.0.0/styles/SBSansText.0.2.0.css" />
    <link rel="stylesheet" href="https://cdn-app.sberdevices.ru/shared-static/0.0.0/styles/SBSansDisplay.0.2.0.css" />
    
    <style>
    /* ==========================================================================
       1. Глобальная дизайн-система Сбера (Plasma Tokens)
       ========================================================================== */
    :root {
        --sber-font-text: 'SB Sans Text', 'SBSansText', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        --sber-font-display: 'SB Sans Display', 'SBSansDisplay', 'SB Sans Text', sans-serif;
        
        --sber-green: #21A038;
        --sber-green-hover: #1FA338;
        --sber-green-dark: #0E7425;
        --sber-green-light: #EBF8EE;
        --sber-green-gradient: linear-gradient(108deg, #0e8528 0%, #21a038 52%, #18d24e 100%);
        --sber-emerald-gradient: linear-gradient(135deg, #083318 0%, #0d5423 50%, #177a33 100%);
        
        --sber-bg: #F4F6F9;
        --sber-surface: #FFFFFF;
        --sber-card-shadow: 0 4px 24px rgba(0, 0, 0, 0.05);
        --sber-card-shadow-hover: 0 8px 32px rgba(33, 160, 56, 0.12);
        
        --sber-text-primary: #121922;
        --sber-text-secondary: #687182;
        --sber-text-muted: #8E98A8;
        --sber-border: rgba(0, 0, 0, 0.07);
        --sber-radius-lg: 20px;
        --sber-radius-md: 14px;
        --sber-radius-pill: 9999px;
    }

    /* Базовый шрифт и фон страницы */
    html, body, [class*="css"], .stApp {
        font-family: var(--sber-font-text) !important;
        background-color: var(--sber-bg) !important;
        color: var(--sber-text-primary) !important;
        letter-spacing: -0.01em;
    }

    /* Заголовки шрифтом SB Sans Display */
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--sber-font-display) !important;
        letter-spacing: -0.025em !important;
        color: var(--sber-text-primary) !important;
        font-weight: 700 !important;
    }

    /* ==========================================================================
       2. Фирменная шапка Сбера (Sber Header)
       ========================================================================== */
    .sber-top-navbar {
        background: #FFFFFF;
        border-radius: var(--sber-radius-lg);
        padding: 16px 28px;
        margin-bottom: 24px;
        box-shadow: 0 2px 16px rgba(0, 0, 0, 0.04);
        border: 1px solid var(--sber-border);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        flex-wrap: wrap;
    }
    .sber-brand-group {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .sber-brand-divider {
        width: 1px;
        height: 32px;
        background: #E5E8ED;
    }
    .sber-service-title {
        font-family: var(--sber-font-display);
        font-size: 19px;
        font-weight: 700;
        color: var(--sber-text-primary);
        margin: 0;
        line-height: 1.2;
    }
    .sber-service-subtitle {
        font-family: var(--sber-font-text);
        font-size: 13px;
        color: var(--sber-text-secondary);
        margin: 2px 0 0 0;
    }
    .sber-telemetry-pills {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    .sber-pill-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: var(--sber-radius-pill);
        font-size: 12.5px;
        font-weight: 600;
        border: 1px solid transparent;
        background: #F4F6F9;
        color: var(--sber-text-primary);
    }
    .sber-pill-status.active {
        background: var(--sber-green-light);
        color: var(--sber-green-dark);
        border-color: rgba(33, 160, 56, 0.25);
    }
    .sber-pulse-dot {
        width: 8px;
        height: 8px;
        background-color: var(--sber-green);
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(33, 160, 56, 0.7);
        animation: sberPulse 2s infinite;
    }
    @keyframes sberPulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(33, 160, 56, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(33, 160, 56, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(33, 160, 56, 0); }
    }

    /* ==========================================================================
       3. Виджеты KPI в стиле СберБанк Онлайн
       ========================================================================== */
    .sber-kpi-card {
        background: var(--sber-surface);
        border-radius: var(--sber-radius-lg);
        padding: 20px;
        border: 1px solid var(--sber-border);
        box-shadow: var(--sber-card-shadow);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .sber-kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--sber-card-shadow-hover);
        border-color: rgba(33, 160, 56, 0.3);
    }
    .sber-kpi-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    .sber-kpi-icon-wrap {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--sber-green-light);
    }
    .sber-kpi-label {
        font-size: 13px;
        font-weight: 500;
        color: var(--sber-text-secondary);
        margin: 0;
    }
    .sber-kpi-value {
        font-family: var(--sber-font-display);
        font-size: 28px;
        font-weight: 700;
        color: var(--sber-text-primary);
        letter-spacing: -0.03em;
        margin: 6px 0 8px 0;
        line-height: 1.1;
    }
    .sber-kpi-subtext {
        font-size: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
        font-weight: 500;
    }

    /* ==========================================================================
       4. Карточки решений Калиты (Decision Engine)
       ========================================================================== */
    .sber-decision-box {
        background: var(--sber-surface);
        border-radius: var(--sber-radius-lg);
        padding: 22px 26px;
        border: 1px solid var(--sber-border);
        box-shadow: var(--sber-card-shadow);
        margin: 16px 0 24px 0;
        position: relative;
        overflow: hidden;
    }
    .sber-decision-box::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 6px;
    }
    .sber-decision-box.success::before {
        background: var(--sber-green);
    }
    .sber-decision-box.warning::before {
        background: #F59E0B;
    }
    .sber-decision-box.danger::before {
        background: #EF4444;
    }
    .sber-decision-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    .sber-decision-title {
        font-family: var(--sber-font-display);
        font-size: 20px;
        font-weight: 700;
        color: var(--sber-text-primary);
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* ==========================================================================
       5. Стилизация Streamlit элементов под Сбер
       ========================================================================== */
    /* Табы */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #E8EDF2;
        padding: 6px;
        border-radius: var(--sber-radius-pill);
        border-bottom: none !important;
        margin-bottom: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--sber-radius-pill) !important;
        padding: 10px 22px !important;
        font-family: var(--sber-font-text) !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        color: var(--sber-text-secondary) !important;
        background: transparent !important;
        border: none !important;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: var(--sber-text-primary) !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Кнопки в стиле Сбера (Pill Buttons) */
    .stButton > button, .stDownloadButton > button {
        background: var(--sber-green) !important;
        background: var(--sber-green-gradient) !important;
        color: #FFFFFF !important;
        font-family: var(--sber-font-text) !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        border-radius: var(--sber-radius-pill) !important;
        border: none !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 14px rgba(33, 160, 56, 0.28) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(33, 160, 56, 0.38) !important;
    }
    .stButton > button:active, .stDownloadButton > button:active {
        transform: translateY(1px) !important;
    }

    /* Прогресс-бар Сбер */
    .stProgress > div > div > div > div {
        background: var(--sber-green-gradient) !important;
        border-radius: var(--sber-radius-pill) !important;
    }
    .stProgress > div > div {
        background-color: #E2E8F0 !important;
        border-radius: var(--sber-radius-pill) !important;
        height: 10px !important;
    }

    /* Сайдбар */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid var(--sber-border) !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--sber-text-primary) !important;
    }

    /* Селекты и поля ввода */
    [data-baseweb="select"] > div {
        border-radius: var(--sber-radius-md) !important;
        border-color: var(--sber-border) !important;
    }
    [data-baseweb="select"] > div:focus-within {
        border-color: var(--sber-green) !important;
        box-shadow: 0 0 0 3px rgba(33, 160, 56, 0.15) !important;
    }

    /* Бейджи */
    .sber-badge-green {
        background: var(--sber-green-light);
        color: var(--sber-green-dark);
        border: 1px solid rgba(33, 160, 56, 0.25);
        padding: 4px 12px;
        border-radius: var(--sber-radius-pill);
        font-size: 12px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .sber-badge-yellow {
        background: #FEF9C3;
        color: #854D0E;
        border: 1px solid rgba(234, 179, 8, 0.25);
        padding: 4px 12px;
        border-radius: var(--sber-radius-pill);
        font-size: 12px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .sber-badge-red {
        background: #FEE2E2;
        color: #B91C1C;
        border: 1px solid rgba(239, 68, 68, 0.25);
        padding: 4px 12px;
        border-radius: var(--sber-radius-pill);
        font-size: 12px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Контейнер карточек */
    .sber-white-card {
        background: #FFFFFF;
        border-radius: var(--sber-radius-lg);
        padding: 24px;
        border: 1px solid var(--sber-border);
        box-shadow: var(--sber-card-shadow);
        margin-bottom: 20px;
    }
    
    /* Сбер Таблицы */
    .sber-table-v2 {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 13.5px;
        margin-top: 12px;
        border-radius: var(--sber-radius-md);
        overflow: hidden;
        border: 1px solid var(--sber-border);
    }
    .sber-table-v2 th {
        background-color: #F8FAFC;
        color: var(--sber-text-secondary);
        font-weight: 600;
        padding: 12px 16px;
        text-align: left;
        border-bottom: 1px solid var(--sber-border);
    }
    .sber-table-v2 td {
        padding: 12px 16px;
        border-bottom: 1px solid var(--sber-border);
        color: var(--sber-text-primary);
        background: #FFFFFF;
    }
    .sber-table-v2 tr:last-child td {
        border-bottom: none;
    }
    .sber-table-v2 tr:hover td {
        background-color: #F8FDF9;
    }
    </style>
    """


def render_sber_header(active_objects_count: int = 13, rtk_status: str = "FIX Q=16") -> str:
    """Генерирует HTML-шапку в фирменном стиле sberbank.ru."""
    return f"""
    <div class="sber-top-navbar">
        <div class="sber-brand-group">
            {SBER_LOGO_SVG}
            <div class="sber-brand-divider"></div>
            <div>
                <h1 class="sber-service-title">Центр мониторинга залогов ИЖС</h1>
                <p class="sber-service-subtitle">Сибирский банк ПАО Сбербанк • Инспекция БПЛА & АС «Калита»</p>
            </div>
        </div>
        <div class="sber-telemetry-pills">
            <div class="sber-pill-status active">
                <span class="sber-pulse-dot"></span>
                <span>DJI RTK {rtk_status}</span>
            </div>
            <div class="sber-pill-status">
                <span>⚡ АС «Калита» Онлайн</span>
            </div>
            <div class="sber-pill-status" style="background:#EBF5FF; color:#1D4ED8;">
                <span>🎯 Роль #3 (Инспектор-дизайнер)</span>
            </div>
        </div>
    </div>
    """


def render_kpi_card(title: str, value: str, subtext: str, icon_type: str = "drone", is_alert: bool = False, is_success: bool = False) -> str:
    """Генерирует виджет карточки KPI в стиле СберБанк Онлайн."""
    icon_svg = SVG_ICONS.get(icon_type, SVG_ICONS["drone"])
    sub_color = "#15803D" if is_success else ("#B91C1C" if is_alert else "#687182")
    icon_bg = "#FEE2E2" if is_alert else ("#EAF8ED" if is_success else "#F4F6F9")
    
    return f"""
    <div class="sber-kpi-card">
        <div class="sber-kpi-top">
            <span class="sber-kpi-label">{title}</span>
            <div class="sber-kpi-icon-wrap" style="background:{icon_bg};">
                {icon_svg}
            </div>
        </div>
        <div>
            <div class="sber-kpi-value">{value}</div>
            <div class="sber-kpi-subtext" style="color:{sub_color};">
                {subtext}
            </div>
        </div>
    </div>
    """
