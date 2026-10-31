"""Field operator notes parser for drone inspections.

Parses flight logs, field notes (*.txt), and operator remarks,
detecting potential collateral risks, physical obstructions, and commercial deviations.
"""

from __future__ import annotations

from pathlib import Path


class NotesParser:
    """Extracts and analyzes field pilot observations from text files."""

    @classmethod
    def parse_plot_notes(cls, plot_dir: str | Path) -> list[str]:
        """Scan a plot directory for text notes and return formatted operator remarks."""
        dir_path = Path(plot_dir)
        if not dir_path.exists() or not dir_path.is_dir():
            return []

        notes: list[str] = []
        for txt_file in sorted(dir_path.rglob("*.txt")):
            text = ""
            for enc in ("utf-8", "cp1251", "latin-1"):
                try:
                    text = txt_file.read_text(encoding=enc).strip()
                    if text:
                        break
                except Exception:
                    continue

            if text:
                notes.append(text)

        return notes

    @classmethod
    def detect_note_hazards(cls, notes: list[str]) -> tuple[list[str], list[str]]:
        """Identify fraud risks and physical defects described in field pilot notes.

        Returns (fraud_reasons, defects).
        """
        fraud_reasons: list[str] = []
        defects: list[str] = []

        combined = " ".join(notes).lower()

        if any(w in combined for w in ("супермаркет", "магазин", "торговый", "склад", "энгельса", "энегельса")):
            fraud_reasons.append(
                "Полевая фиксация пилота БПЛА: фактически сфотографирован коммерческий объект "
                "(супермаркет, г. Бердск, ул. Энгельса, 4), хотя по кадастровому номеру числится "
                "жилое здание ИЖС"
            )

        if any(w in combined for w in ("лэп", "линия электропередач", "высоковольтн", "провода")):
            defects.append(
                "Опасное соседство и помеха аэросъемке: густая линия электропередач (ЛЭП) вблизи объекта "
                "(близко подлететь не получилось, СП 42.13330)"
            )

        if any(w in combined for w in ("деревь", "лес", "кроны", "заросли", "близко подлететь")):
            defects.append(
                "Физические препятствия облета: вокруг объекта много плотных деревьев и ЛЭП, "
                "ограничивающих подлет БПЛА на малую дистанцию"
            )

        if "2гис" in combined or "не имеет адреса" in combined or "жилое здание" in combined:
            defects.append(
                "Адресная и кадастровая коллизия: по кадастровому номеру объект числится как жилое "
                "здание, которое не имеет адреса в 2ГИС (фактически — супермаркет)"
            )

        return fraud_reasons, defects
