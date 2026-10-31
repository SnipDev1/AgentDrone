"""Unit tests for Sberbank Design System and UI theme components."""

from __future__ import annotations

import unittest

from src.ui.sber_theme import (
    SBER_LOGO_SVG,
    SVG_ICONS,
    get_sber_css,
    render_kpi_card,
    render_sber_header,
)


class TestSberTheme(unittest.TestCase):
    """Test suite for Sber UI components and Plasma CSS styling."""

    def test_sber_css_contains_fonts_and_tokens(self):
        css = get_sber_css()
        self.assertIn("SBSansText.0.2.0.css", css)
        self.assertIn("SBSansDisplay.0.2.0.css", css)
        self.assertIn("--sber-green: #21A038", css)
        self.assertIn("--sber-bg: #F4F6F9", css)
        self.assertIn(".sber-kpi-card", css)
        self.assertIn(".sber-decision-box", css)
        self.assertIn(".sber-badge-green", css)
        self.assertIn(".sber-table-v2", css)

    def test_sber_logo_svg_validity(self):
        self.assertIn("<svg", SBER_LOGO_SVG)
        self.assertIn("СБЕР", SBER_LOGO_SVG)
        self.assertIn("sberGrad", SBER_LOGO_SVG)

    def test_render_sber_header(self):
        header = render_sber_header(active_objects_count=13, rtk_status="FIX Q=16")
        self.assertIn("sber-top-navbar", header)
        self.assertIn("AgentDrone", header)
        self.assertIn("Объектов: 13", header)
        self.assertNotIn("DJI RTK FIX Q=16", header)
        self.assertNotIn("АС «Калита» Онлайн", header)

    def test_render_kpi_card(self):
        card = render_kpi_card(
            title="Готовность объекта",
            value="85.4%",
            subtext="Стадия 100%",
            icon_type="chart",
            is_success=True,
        )
        self.assertIn("sber-kpi-card", card)
        self.assertIn("Готовность объекта", card)
        self.assertIn("85.4%", card)
        self.assertIn("Стадия 100%", card)
        self.assertIn("<svg", card)

    def test_svg_icons_palette(self):
        for icon_key in ["drone", "money", "chart", "shield_alert", "check_circle", "satellite"]:
            self.assertIn(icon_key, SVG_ICONS)
            self.assertIn("<svg", SVG_ICONS[icon_key])


if __name__ == "__main__":
    unittest.main()
