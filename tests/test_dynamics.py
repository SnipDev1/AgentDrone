"""Unit tests for multi-temporal construction dynamics and AI Agent audit."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.engine.dynamics_analyzer import (
    DynamicsAnalyzer,
    extract_date_from_image,
    group_images_by_date,
)
from src.gigachat_service.service import GigaChatInspectorService
from src.models.dynamics import FlightSnapshot
from src.models.schemas import WallMaterial


class TestConstructionDynamics(unittest.TestCase):
    """Test suite for construction dynamics calculations and GigaChat AI agent audit."""

    def test_extract_date_from_image_formats(self):
        """Verify extraction of dates across DJI standard and ISO filenames."""
        p_dji = Path("dataset/5419160202125/DJI_20260807184615_0002_D.JPG")
        self.assertEqual(extract_date_from_image(p_dji), "2026-08-07")

        p_iso = Path("/tmp/flights/shot_2026-05-15_ortho.jpg")
        self.assertEqual(extract_date_from_image(p_iso), "2026-05-15")

        p_parent = Path("/tmp/2026-04-10/DJI_0001.JPG")
        self.assertEqual(extract_date_from_image(p_parent), "2026-04-10")

    def test_group_images_by_date(self):
        """Verify images are partitioned chronologically by calendar date."""
        imgs = [
            Path("/data/DJI_20260807120000_0001_D.JPG"),
            Path("/data/DJI_20260515120000_0001_D.JPG"),
            Path("/data/DJI_20260807120500_0002_D.JPG"),
        ]
        grouped = group_images_by_date(imgs)
        dates = list(grouped.keys())
        self.assertEqual(dates, ["2026-05-15", "2026-08-07"])
        self.assertEqual(len(grouped["2026-05-15"]), 1)
        self.assertEqual(len(grouped["2026-08-07"]), 2)

    def test_compute_dynamics_progression(self):
        """Verify calculation of readiness delta, elapsed days, and monthly pace."""
        snap1 = FlightSnapshot(
            flight_id="flight_20260515",
            flight_date="2026-05-15",
            readiness_pct=15.0,
            stage=0,
            stage_title="Стадия 0%",
            has_foundation=True,
            has_walls=False,
            has_roof=False,
            has_windows_doors=False,
            detected_material=WallMaterial.BRICK,
            shots_count=6,
        )
        snap2 = FlightSnapshot(
            flight_id="flight_20260807",
            flight_date="2026-08-07",
            readiness_pct=88.0,
            stage=100,
            stage_title="Стадия 100%",
            has_foundation=True,
            has_walls=True,
            has_roof=True,
            has_windows_doors=True,
            detected_material=WallMaterial.BRICK,
            shots_count=6,
        )

        dyn = DynamicsAnalyzer.compute_dynamics(
            cadastral_id="54:19:160202:125",
            snapshots=[snap1, snap2],
        )

        self.assertTrue(dyn.has_multiple_dates)
        self.assertEqual(dyn.delta_readiness_pct, 73.0)
        self.assertEqual(dyn.delta_days, 84)
        self.assertGreater(dyn.pace_pct_per_month, 20.0)
        self.assertIn("Высокий темп", dyn.pace_status)
        self.assertTrue(any("стеновой комплект" in p.lower() for p in dyn.elements_progress))
        self.assertTrue(any("кров" in p.lower() for p in dyn.elements_progress))
        self.assertTrue(any("тепловой контур" in p.lower() for p in dyn.elements_progress))

    def test_compute_dynamics_stalled_construction(self):
        """Verify detection of frozen / stalled construction (0% delta over 60 days)."""
        snap1 = FlightSnapshot(
            flight_id="f1",
            flight_date="2026-05-01",
            readiness_pct=15.0,
            stage=0,
            has_foundation=True,
        )
        snap2 = FlightSnapshot(
            flight_id="f2",
            flight_date="2026-07-01",
            readiness_pct=15.0,
            stage=0,
            has_foundation=True,
        )

        dyn = DynamicsAnalyzer.compute_dynamics(
            cadastral_id="54:19:160202:125",
            snapshots=[snap1, snap2],
        )

        self.assertEqual(dyn.delta_readiness_pct, 0.0)
        self.assertIn("Заморозка", dyn.pace_status)

    def test_save_and_load_history_persistence(self):
        """Verify flight history saves and loads from JSON storage."""
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            tmp_path = Path(tmp.name)
            cad_id = "54:19:999999:999"

            snap = FlightSnapshot(
                flight_id="f_test",
                flight_date="2026-06-20",
                readiness_pct=50.0,
                stage=50,
            )

            DynamicsAnalyzer.save_history(cad_id, snap, history_file=tmp_path)
            loaded = DynamicsAnalyzer.load_history(cad_id, history_file=tmp_path)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].flight_date, "2026-06-20")
            self.assertEqual(loaded[0].readiness_pct, 50.0)

    def test_gigachat_dynamics_analysis_offline_and_prompt(self):
        """Verify GigaChat dynamics prompt and deterministic offline summary generation."""
        service = GigaChatInspectorService()
        snap1 = FlightSnapshot(
            flight_id="f1",
            flight_date="2026-05-15",
            readiness_pct=15.0,
            stage=0,
            has_foundation=True,
        )
        snap2 = FlightSnapshot(
            flight_id="f2",
            flight_date="2026-08-07",
            readiness_pct=88.0,
            stage=100,
            has_foundation=True,
            has_walls=True,
            has_roof=True,
            has_windows_doors=True,
        )
        dyn = DynamicsAnalyzer.compute_dynamics("54:19:160202:125", [snap1, snap2])

        # 1. Test deterministic offline template
        offline_summary = service._generate_offline_dynamics_summary(dyn)
        self.assertIn("ЭКСПЕРТНОЕ ЗАКЛЮЧЕНИЕ", offline_summary)
        self.assertIn("54:19:160202:125", offline_summary)
        self.assertIn("2026-05-15", offline_summary)
        self.assertIn("2026-08-07", offline_summary)
        self.assertIn("залоговой службы", offline_summary)

        # 2. Test universal call (online or fallback)
        summary = service.analyze_construction_dynamics(dyn)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 50)
        self.assertTrue("54:19:160202:125" in summary or "залог" in summary.lower() or "готовност" in summary.lower())
