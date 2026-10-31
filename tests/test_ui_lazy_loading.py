"""Unit tests for on-demand / lazy plot updating in the Streamlit UI.

Verifies that:
1. Collateral plots are NOT processed in bulk upfront.
2. Uninspected plots produce lightweight registry summaries without running heavy CV.
3. Inspected plots accurately merge full CV & Decision Engine analysis.
4. Portfolio summary correctly distinguishes visited vs unvisited plots.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.ingestion.credit_registry import CreditRegistry
from src.ui.app import (
    build_plot_summary,
    get_available_plot_ids,
    get_portfolio_summary,
)


class TestUILazyLoading(unittest.TestCase):
    """Test suite for on-demand / lazy plot inspection and summary assembly."""

    def setUp(self):
        self.plot_ids = get_available_plot_ids()
        self.assertGreater(len(self.plot_ids), 0, "Plot IDs must be available")
        self.sample_id = self.plot_ids[0]

    def test_uninspected_plot_summary_lightweight(self):
        """Unvisited plot should return basic registry data with is_inspected=False and no CV."""
        summary = build_plot_summary(self.sample_id, inspected=None)

        self.assertFalse(summary["is_inspected"])
        self.assertIsNone(summary["readiness_pct"])
        self.assertEqual(summary["stage"], "—")
        self.assertEqual(summary["strategy_code"], "ОЖИДАЕТ")
        self.assertEqual(summary["risk_level"], "PENDING")
        self.assertFalse(summary["fraud_flag"])

        # Should still have valid financial and cadastral metadata
        credit = CreditRegistry.get_credit_object(self.sample_id)
        if credit:
            self.assertEqual(summary["borrower"], credit.borrower_name)
            self.assertEqual(summary["credit_amount"], credit.credit_amount_rub)

    def test_inspected_plot_summary_contains_full_cv_results(self):
        """Visited plot should contain full CV and Decision Engine parameters."""
        mock_cv = MagicMock()
        mock_cv.cadastral_id = "54:07:055001:178"
        mock_cv.readiness_pct = 75.5
        mock_cv.readiness_stage = 50
        mock_cv.detected_material.value = "Кирпич"
        mock_cv.fraud_risk_flag = False

        mock_credit = MagicMock()
        mock_credit.borrower_name = "Иванов Иван Иванович"
        mock_credit.address = "Тестовый адрес"
        mock_credit.credit_amount_rub = 5_000_000.0
        mock_credit.days_overdue = 0

        mock_decision = MagicMock()
        mock_decision.strategy.value = "Выдача очередного транша"
        mock_decision.strategy_code = "STRAT-01"
        mock_decision.risk_level = "LOW"
        mock_decision.kalita_status = "Транш согласован"

        summary = build_plot_summary(
            self.sample_id,
            inspected=(mock_cv, mock_credit, mock_decision),
        )

        self.assertTrue(summary["is_inspected"])
        self.assertEqual(summary["readiness_pct"], 75.5)
        self.assertEqual(summary["stage"], 50)
        self.assertEqual(summary["material"], "Кирпич")
        self.assertEqual(summary["strategy_code"], "STRAT-01")
        self.assertEqual(summary["risk_level"], "LOW")
        self.assertEqual(summary["kalita_status"], "Транш согласован")
        self.assertEqual(summary["borrower"], "Иванов Иван Иванович")

    def test_portfolio_summary_only_marks_visited_plots(self):
        """When user navigates to 1 plot, only that 1 plot is inspected in portfolio data."""
        mock_cv = MagicMock()
        mock_cv.cadastral_id = self.sample_id
        mock_cv.readiness_pct = 45.0
        mock_cv.readiness_stage = 50
        mock_cv.detected_material.value = "Газобетон"
        mock_cv.fraud_risk_flag = False

        mock_credit = MagicMock()
        mock_credit.borrower_name = "Тестовый Заемщик"
        mock_credit.address = "Адрес"
        mock_credit.credit_amount_rub = 4_000_000.0
        mock_credit.days_overdue = 0

        mock_decision = MagicMock()
        mock_decision.strategy.value = "Реструктуризация"
        mock_decision.strategy_code = "STRAT-02"
        mock_decision.risk_level = "MEDIUM"
        mock_decision.kalita_status = "На согласовании"

        inspected_cache = {
            self.sample_id: (mock_cv, mock_credit, mock_decision),
        }

        portfolio = get_portfolio_summary(self.plot_ids, inspected_cache)
        self.assertEqual(len(portfolio), len(self.plot_ids))

        # Sample plot must be inspected
        sample_row = next(r for r in portfolio if r["raw_id"] == self.sample_id)
        self.assertTrue(sample_row["is_inspected"])
        self.assertEqual(sample_row["readiness_pct"], 45.0)
        self.assertEqual(sample_row["strategy_code"], "STRAT-02")

        # All other plots must be uninspected (lazy loaded)
        other_rows = [r for r in portfolio if r["raw_id"] != self.sample_id]
        self.assertGreater(len(other_rows), 0)
        for row in other_rows:
            self.assertFalse(row["is_inspected"])
            self.assertIsNone(row["readiness_pct"])
            self.assertEqual(row["strategy_code"], "ОЖИДАЕТ")

    @patch("src.cv.pipeline.CVPipeline.analyze_plot")
    def test_unvisited_plots_do_not_trigger_cv_pipeline(self, mock_cv_analyze):
        """Assembling portfolio summary must never invoke CVPipeline for unvisited plots."""
        # Ensure cache has no inspected plots
        inspected_cache = {}

        summary = get_portfolio_summary(self.plot_ids, inspected_cache)

        self.assertEqual(len(summary), len(self.plot_ids))
        # CVPipeline.analyze_plot should NOT have been called even once!
        mock_cv_analyze.assert_not_called()


if __name__ == "__main__":
    unittest.main()
