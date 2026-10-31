"""Report generator for Siberian Bank Collateral Inspection Act (Приложение 1).

Renders official Jinja2 HTML reports and handles export to file.
"""

from pathlib import Path
from typing import Optional

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    _HAS_JINJA2 = True
except ImportError:
    _HAS_JINJA2 = False

from ..models import ActReportData

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


class ActGenerator:
    """Generates official inspection act documents in HTML/PDF."""

    def __init__(self, templates_dir: Optional[Path | str] = None):
        self.templates_dir = Path(templates_dir) if templates_dir else TEMPLATES_DIR
        if _HAS_JINJA2:
            self.env = Environment(
                loader=FileSystemLoader(str(self.templates_dir)),
                autoescape=select_autoescape(["html", "xml"]),
            )
            self.template = self.env.get_template("act_template.html")
        else:
            self.env = None
            self.template = None

    def render_html(self, data: ActReportData) -> str:
        """Render ActReportData to full HTML string."""
        if self.template is not None:
            return self.template.render(data=data)
        # Fallback HTML renderer if jinja2 is not installed in bare environment
        c = data.credit
        i = data.inspection
        r = data.recommendation
        elems = i.elements
        defects_html = (
            "".join(f"<li>{d.defect_type}: {d.description}</li>" for d in i.defects)
            if i.defects
            else "<li>Дефекты и отклонения не выявлены</li>"
        )
        photos_html = "".join(
            f'<div class="photo"><img src="{p.file_path}" alt="{p.caption}"/><p>{p.caption}</p></div>'
            for p in data.photos
        )
        reasons_html = "".join(f"<li>{rs}</li>" for rs in r.reasons)
        actions_html = "".join(f"<li>{ac}</li>" for ac in r.action_items)
        return (
            f"<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'>"
            f"<title>Акт проверки наличия и состояния залога № {data.act_number}</title></head>"
            f"<body><h1>АКТ ПРОВЕРКИ ИМУЩЕСТВА (Приложение № 1)</h1>"
            f"<p>Номер акта: {data.act_number} | Дата: {data.act_date} | АС «Калита» | {data.branch_name}</p>"
            f"<h2>1. Сведения о кредитном договоре и заемщике</h2>"
            f"<p>Договор: {c.credit_id} | Заемщик: {c.borrower_name} | Кадастровый номер: {c.cadastral_number}</p>"
            f"<p>Адрес: {c.address} | Сумма: {c.credit_amount:,.2f} | Выдано: {c.current_disbursed:,.2f} | "
            f"Транш: {c.tranche_number} из {c.tranches_total} ({c.next_tranche_amount:,.2f}) | "
            f"Просрочка: {c.days_overdue} дн. | Проектный материал: {c.projected_wall_material.value}</p>"
            f"<h2>2. Результаты аэрофотограмметрического мониторинга БПЛА</h2>"
            f"<p>Готовность: {i.physical_readiness_pct:.1f}% | Стадия: {i.stage.value}% | "
            f"Фактический материал: {i.detected_wall_material.value} | Состояние: {data.condition.value}</p>"
            f"<p>Фундамент: {'Да' if elems.foundation else 'Нет'} | Стены: {'Да' if elems.walls else 'Нет'} | "
            f"Кровля: {'Да' if elems.roof else 'Нет'} | Окна/двери: {'Да' if elems.windows_doors else 'Нет'}</p>"
            f"<ul>{defects_html}</ul>"
            f"<h2>3. Решение и стратегия банка ({r.decision_code})</h2>"
            f"<p>Стратегия: {r.strategy.value} | Уровень риска: {r.risk_level.value}</p>"
            f"<ul>{reasons_html}</ul><ul>{actions_html}</ul>"
            f"<h2>4. Заключение инспектора</h2><p>{r.summary}</p>"
            f"<div>{photos_html}</div>"
            f"<p>Инспектор: {data.inspector_name} ({data.inspector_position})</p></body></html>"
        )

    def save_html(self, data: ActReportData, output_path: Path | str) -> Path:
        """Render and save HTML report to target destination."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        html_content = self.render_html(data)
        out.write_text(html_content, encoding="utf-8")
        return out
