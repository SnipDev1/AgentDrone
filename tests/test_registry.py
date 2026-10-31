"""Unit tests for Credit Registry (DATA-04).

Verifies loading, Pydantic validation, and cadastral mapping of the 13
test credit objects for Siberian Bank.
"""

from pathlib import Path
import pytest
from src.engine.registry import CreditRegistry, get_default_registry
from src.dataset import DroneDataset


def test_registry_loads_all_13_objects():
    """Verify that exactly 13 credit records are loaded and validated."""
    reg = CreditRegistry()
    all_credits = reg.list_all()
    assert len(all_credits) == 13
    assert len(reg) == 13


def test_registry_matches_dataset_plots():
    """Verify that every cadastral plot in dataset has a matching credit in the registry."""
    ds = DroneDataset(Path("dataset/24.09.20206"))
    if not ds.is_available():
        pytest.skip("Dataset root not found, skipping plot matching")

    reg = CreditRegistry()
    dataset_plots = ds.list_plots()
    assert len(dataset_plots) == 13

    for plot_id in dataset_plots:
        credit = reg.get_by_cadastral(plot_id)
        assert credit is not None, f"No credit record found for plot {plot_id}"
        assert credit.clean_cadastral == plot_id


def test_registry_lookups():
    """Verify lookup by both formatted and clean cadastral numbers."""
    reg = get_default_registry()

    # Formatted
    c1 = reg.get_by_cadastral("54:07:055001:178")
    assert c1 is not None
    assert c1.credit_id == "CR-SIB-2026-001"
    assert c1.borrower_name == "Сидоров Андрей Владимирович"

    # Cleaned digits
    c2 = reg.get_by_cadastral("5407055001178")
    assert c2 is not None
    assert c2.credit_id == c1.credit_id

    # By credit ID
    c3 = reg.get_by_credit_id("CR-SIB-2026-011")
    assert c3 is not None
    assert "Бердск" in c3.address

    # Non-existent
    assert reg.get_by_cadastral("99999999999") is None


def test_registry_overdue_and_financials():
    """Verify realistic financial amounts and overdue filtering."""
    reg = get_default_registry()
    overdue = reg.get_overdue_loans()
    assert len(overdue) >= 2  # We configured realistic overdue cases

    for c in reg.list_all():
        assert c.credit_amount >= 1000000.0  # Real mortgage sums
        assert c.construction_deadline > c.credit_issue_date
