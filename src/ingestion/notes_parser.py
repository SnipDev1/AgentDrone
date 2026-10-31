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
        for txt_file in dir_path.glob("*.txt"):
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

        if any(w in combined for w in ("супермаркет", "магазин", "торговый", "склад")):
            fraud_reasons.append(
                "Полевая фиксация пилота БПЛА: возведен коммерческий объект (супермаркет) "
                "вместо жилого дома по ИЖС"
            )

        if any(w in combined for w in ("лэп", "линия электропередач", "высоковольтн")):
            defects.append(
                "Опасное соседство: густая линия электропередач (ЛЭП) в охранной зоне ЗУ "
                "(СП 42.13330)"
            )

        if any(w in combined for w in ("деревь", "лес", "кроны", "заросли")):
            defects.append(
                "Нарушение противопожарного разрыва: плотные кроны деревьев вблизи строения"
            )

        if "2гис" in combined and "не имеет адреса" in combined:
            defects.append(
                "Адресная коллизия: объект отсутствует в геоинформационной системе 2ГИС"
            )

        return fraud_reasons, defects
