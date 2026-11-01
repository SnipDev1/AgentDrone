"""Construction Dynamics Analyzer for AgentDrone.

Extracts multi-temporal flight epochs, compares construction progress across dates,
computes rate of work (% per month), and prepares payloads for the GigaChat AI Agent.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

from src.models.dynamics import ConstructionDynamics, FlightSnapshot
from src.models.schemas import WallMaterial


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_HISTORY_FILE = PROJECT_ROOT / "data" / "flight_history.json"
_RELATED_CV_CACHE: dict[str, Any] = {}


def extract_date_from_image(path: Path) -> str:
    """Extract capture date (YYYY-MM-DD) from image filename, parent dir, or file mtime."""
    name = path.name

    # 1. Match DJI standard filename format: DJI_YYYYMMDD...
    m = re.search(r"DJI_(\d{4})(\d{2})(\d{2})", name, re.IGNORECASE)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # 2. Match standard ISO dates in filename: YYYY-MM-DD or YYYY_MM_DD
    m_iso = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", name)
    if m_iso:
        return f"{m_iso.group(1)}-{m_iso.group(2)}-{m_iso.group(3)}"

    # 3. Check immediate parent folder name (e.g. "2026-05-15" or "flight_20260515")
    parent_name = path.parent.name
    m_p = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", parent_name)
    if m_p:
        return f"{m_p.group(1)}-{m_p.group(2)}-{m_p.group(3)}"

    # 4. Fallback to file modification time
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def group_images_by_date(images: list[Path]) -> dict[str, list[Path]]:
    """Group drone photos by calendar date (YYYY-MM-DD) in chronological order."""
    grouped: dict[str, list[Path]] = {}
    for p in images:
        d = extract_date_from_image(p)
        grouped.setdefault(d, []).append(p)
    return dict(sorted(grouped.items(), key=lambda kv: kv[0]))


class DynamicsAnalyzer:
    """Service for computing multi-temporal construction dynamics and flight history."""

    @classmethod
    def load_history(cls, cadastral_id: str, history_file: Path | None = None) -> list[FlightSnapshot]:
        """Load stored historical flight snapshots for a cadastral parcel."""
        h_file = history_file or DEFAULT_HISTORY_FILE
        if not h_file.exists():
            return []
        try:
            data = json.loads(h_file.read_text(encoding="utf-8"))
            records = data.get(cadastral_id, [])
            return [FlightSnapshot(**r) for r in records]
        except Exception:
            return []

    @classmethod
    def save_history(
        cls,
        cadastral_id: str,
        snapshot: FlightSnapshot,
        history_file: Path | None = None,
    ) -> None:
        """Persist a flight snapshot into the JSON history registry."""
        h_file = history_file or DEFAULT_HISTORY_FILE
        h_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(h_file.read_text(encoding="utf-8")) if h_file.exists() else {}
        except Exception:
            data = {}

        existing_records = data.get(cadastral_id, [])
        # Avoid duplicate dates
        filtered = [r for r in existing_records if r.get("flight_date") != snapshot.flight_date]
        filtered.append(snapshot.model_dump())
        filtered.sort(key=lambda r: r.get("flight_date", ""))
        data[cadastral_id] = filtered

        try:
            h_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    @classmethod
    def build_snapshots_for_plot(
        cls,
        cadastral_id: str,
        plot_images: list[Path],
        current_cv_result: Any,
        include_baseline_if_single: bool = False,
        credit_obj: Any = None,
        plot_id: str | None = None,
    ) -> list[FlightSnapshot]:
        """Extract all chronological flight snapshots for a cadastral plot."""
        from src.dataset_manager import ParcelManager, sanitize_plot_id
        from src.ingestion.cadastral_registry import normalize_cadastral_id

        date_groups = group_images_by_date(plot_images)
        snapshots: list[FlightSnapshot] = []

        # 1. Base snapshot from current inspection folder
        cur_date = getattr(current_cv_result, "inspection_date", None)
        if not cur_date:
            cur_date = next(iter(date_groups.keys())) if date_groups else datetime.now().strftime("%Y-%m-%d")

        se = getattr(current_cv_result, "structural_elements", None) if current_cv_result else None
        cur_snapshot = FlightSnapshot(
            flight_id=f"flight_{cur_date.replace('-', '')}",
            flight_date=cur_date,
            readiness_pct=float(getattr(current_cv_result, "readiness_pct", 88.0)) if current_cv_result else 88.0,
            stage=int(getattr(current_cv_result, "readiness_stage", 100)) if current_cv_result else 100,
            stage_title=str(getattr(current_cv_result, "stage_title", "Стадия 100%")) if current_cv_result else "Стадия 100%",
            has_foundation=bool(getattr(se, "has_foundation", True)) if se else True,
            has_walls=bool(getattr(se, "has_walls", True)) if se else True,
            has_roof=bool(getattr(se, "has_roof", True)) if se else True,
            has_windows_doors=bool(getattr(se, "has_windows_doors", True)) if se else True,
            detected_material=getattr(current_cv_result, "detected_material", WallMaterial.BRICK) if current_cv_result else WallMaterial.BRICK,
            shots_count=len(plot_images),
            shots_paths=[str(p) for p in plot_images],
            defects=list(getattr(current_cv_result, "defects_detected", [])) if current_cv_result else [],
        )
        snapshots.append(cur_snapshot)

        # 2. Check if the current folder itself has multiple flight dates
        if len(date_groups) > 1:
            for d, imgs in date_groups.items():
                if d == cur_date:
                    continue
                snapshots.append(
                    FlightSnapshot(
                        flight_id=f"flight_{d.replace('-', '')}",
                        flight_date=d,
                        readiness_pct=float(getattr(current_cv_result, "readiness_pct", 88.0)),
                        stage=int(getattr(current_cv_result, "readiness_stage", 100)),
                        stage_title=str(getattr(current_cv_result, "stage_title", "Стадия 100%")),
                        has_foundation=bool(getattr(se, "has_foundation", True)) if se else True,
                        has_walls=bool(getattr(se, "has_walls", True)) if se else True,
                        has_roof=bool(getattr(se, "has_roof", True)) if se else True,
                        has_windows_doors=bool(getattr(se, "has_windows_doors", True)) if se else True,
                        detected_material=getattr(current_cv_result, "detected_material", WallMaterial.BRICK) if current_cv_result else WallMaterial.BRICK,
                        shots_count=len(imgs),
                        shots_paths=[str(p) for p in imgs],
                    )
                )

        # 3. Discover sibling flight folders on the server for the same cadastral plot
        target_pid = plot_id or cadastral_id
        target_norm = normalize_cadastral_id(target_pid)
        target_clean = sanitize_plot_id(target_pid)
        target_prefix = re.split(r"[_\- ]", str(target_pid).strip())[0]

        try:
            all_plots = ParcelManager.list_plots()
            for p_name in all_plots:
                if p_name == target_pid:
                    continue
                p_norm = normalize_cadastral_id(p_name)
                p_clean = sanitize_plot_id(p_name)
                p_prefix = re.split(r"[_\- ]", str(p_name).strip())[0]
                if (
                    p_norm == target_norm
                    or p_prefix == target_prefix
                    or p_clean == target_clean
                ):
                    # Found sibling parcel flight (e.g. 5419160202125 and 5419160202125_2027)
                    if p_name in _RELATED_CV_CACHE:
                        rel_cv = _RELATED_CV_CACHE[p_name]
                    else:
                        from src.cv.pipeline import CVPipeline
                        try:
                            rel_cv = CVPipeline.analyze_plot(p_name)
                            _RELATED_CV_CACHE[p_name] = rel_cv
                        except Exception:
                            rel_cv = None

                    if rel_cv is not None:
                        rel_date = getattr(rel_cv, "inspection_date", None)
                        if not rel_date:
                            dates_list = ParcelManager.get_plot_flight_dates(p_name)
                            if dates_list:
                                try:
                                    rel_date = datetime.strptime(dates_list[0], "%d.%m.%Y").strftime("%Y-%m-%d")
                                except Exception:
                                    rel_date = dates_list[0]
                        if not rel_date:
                            rel_date = "2026-08-07"

                        rel_se = getattr(rel_cv, "structural_elements", None)
                        snapshots.append(
                            FlightSnapshot(
                                flight_id=f"flight_{rel_date.replace('-', '')}",
                                flight_date=rel_date,
                                readiness_pct=float(getattr(rel_cv, "readiness_pct", 88.0)),
                                stage=int(getattr(rel_cv, "readiness_stage", 100)),
                                stage_title=str(getattr(rel_cv, "stage_title", "Стадия 100%")),
                                has_foundation=bool(getattr(rel_se, "has_foundation", True)) if rel_se else True,
                                has_walls=bool(getattr(rel_se, "has_walls", True)) if rel_se else True,
                                has_roof=bool(getattr(rel_se, "has_roof", True)) if rel_se else True,
                                has_windows_doors=bool(getattr(rel_se, "has_windows_doors", True)) if rel_se else True,
                                detected_material=getattr(rel_cv, "detected_material", WallMaterial.BRICK),
                                shots_count=int(getattr(rel_cv, "analyzed_images_count", 6)),
                                defects=list(getattr(rel_cv, "defects_detected", [])),
                            )
                        )
        except Exception:
            pass

        # 4. Merge persistent history if any
        history = cls.load_history(cadastral_id)
        if history:
            snapshots.extend(history)

        # 5. Deduplicate snapshots by calendar flight date
        seen_dates: dict[str, FlightSnapshot] = {}
        for s in snapshots:
            # Keep earliest or most detailed
            if s.flight_date not in seen_dates:
                seen_dates[s.flight_date] = s
            else:
                existing = seen_dates[s.flight_date]
                if s.readiness_pct > 0 and existing.readiness_pct == 0:
                    seen_dates[s.flight_date] = s

        unique_snapshots = list(seen_dates.values())
        unique_snapshots.sort(key=lambda s: s.flight_date)

        # Optional legacy baseline fallback ONLY if explicitly requested AND len == 1
        if len(unique_snapshots) == 1 and include_baseline_if_single:
            start_date_str = getattr(credit_obj, "construction_start_date", None)
            if not start_date_str or start_date_str == "Нет":
                try:
                    cur_dt = datetime.strptime(unique_snapshots[0].flight_date, "%Y-%m-%d")
                    start_date_str = (cur_dt.replace(month=max(1, cur_dt.month - 3))).strftime("%Y-%m-%d")
                except Exception:
                    start_date_str = "2026-05-15"

            baseline = FlightSnapshot(
                flight_id=f"flight_{start_date_str.replace('-', '')}",
                flight_date=start_date_str,
                readiness_pct=15.0,
                stage=0,
                stage_title="Стадия 0% (Нулевой цикл / Заливка фундамента)",
                has_foundation=True,
                has_walls=False,
                has_roof=False,
                has_windows_doors=False,
                detected_material=unique_snapshots[0].detected_material,
                shots_count=len(plot_images),
            )
            unique_snapshots = [baseline, unique_snapshots[0]]
            unique_snapshots.sort(key=lambda s: s.flight_date)

        return unique_snapshots

    @classmethod
    def compute_dynamics(
        cls,
        cadastral_id: str,
        snapshots: list[FlightSnapshot],
        credit_obj: Any = None,
    ) -> ConstructionDynamics:
        """Calculate quantitative deltas, monthly pace, and progress description across snapshots."""
        if not snapshots:
            return ConstructionDynamics(cadastral_id=cadastral_id)

        if len(snapshots) < 2:
            snap = snapshots[0]
            return ConstructionDynamics(
                cadastral_id=cadastral_id,
                snapshots=snapshots,
                has_multiple_dates=False,
                delta_readiness_pct=0.0,
                delta_days=0,
                pace_pct_per_month=0.0,
                pace_status="Первичный облет (нет базы для расчета темпа)",
                elements_progress=[
                    f"Зафиксирована фактическая готовность: {snap.readiness_pct:.1f}% ({snap.stage_title})"
                ],
            )

        first_snap = snapshots[0]
        last_snap = snapshots[-1]

        delta_pct = round(last_snap.readiness_pct - first_snap.readiness_pct, 1)

        try:
            d1 = datetime.strptime(first_snap.flight_date, "%Y-%m-%d")
            d2 = datetime.strptime(last_snap.flight_date, "%Y-%m-%d")
            delta_days = max((d2 - d1).days, 1)
        except Exception:
            delta_days = 90

        # Calculate progress pace (% per 30 days)
        pace_per_month = round((delta_pct / delta_days) * 30.0, 1) if delta_days > 0 else 0.0

        # Determine pace status
        if delta_pct <= 0.5 and delta_days >= 30:
            pace_status = f"Заморозка (0% за {delta_days} дн. — долгострой)"
        elif pace_per_month >= 15.0:
            pace_status = "Высокий темп (Опережение нормативного графика)"
        elif pace_per_month >= 6.0:
            pace_status = "В графике (Штатный темп СМР)"
        else:
            pace_status = "Умеренный темп (Риск отставания от планового ввода)"

        # Elements progress summary
        progress_items = []
        if not first_snap.has_foundation and last_snap.has_foundation:
            progress_items.append("Завершен нулевой цикл: залит фундамент (+15%)")
        if not first_snap.has_walls and last_snap.has_walls:
            progress_items.append("Возведен стеновой комплект первого и второго уровней (+40%)")
        if not first_snap.has_roof and last_snap.has_roof:
            progress_items.append("Смонтирована стропильная система и кровельное покрытие (+30%)")
        if not first_snap.has_windows_doors and last_snap.has_windows_doors:
            progress_items.append("Закрыт тепловой контур: установлены стеклопакеты и входные группы (+15%)")

        if not progress_items:
            if delta_pct > 0:
                progress_items.append(f"Общий прирост строительной готовности: +{delta_pct}%")
            elif delta_days >= 30 and delta_pct <= 0.5:
                progress_items.append(
                    f"Строительно-монтажные работы за период {first_snap.flight_date} — {last_snap.flight_date} "
                    f"({delta_days} дн.) не проводились: динамика отсутствует (0.0%)."
                )
            else:
                progress_items.append("Строительно-монтажные работы в наблюдаемом интервале не проводились")

        return ConstructionDynamics(
            cadastral_id=cadastral_id,
            snapshots=snapshots,
            has_multiple_dates=True,
            delta_readiness_pct=delta_pct,
            delta_days=delta_days,
            pace_pct_per_month=pace_per_month,
            pace_status=pace_status,
            elements_progress=progress_items,
        )
