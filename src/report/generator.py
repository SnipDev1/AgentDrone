"""Report generator for Siberian Bank Collateral Inspection Act (Приложение 1).

Renders official Jinja2 HTML reports and handles export to file.
"""

from pathlib import Path
from typing import Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..models import ActReportData

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


class ActGenerator:
    """Generates official inspection act documents in HTML/PDF."""

    def __init__(self, templates_dir: Optional[Path | str] = None):
        self.templates_dir = Path(templates_dir) if templates_dir else TEMPLATES_DIR
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        self.template = self.env.get_template("act_template.html")

    def render_html(self, data: ActReportData) -> str:
        """Render ActReportData to full HTML string."""
        return self.template.render(data=data)

    def save_html(self, data: ActReportData, output_path: Path | str) -> Path:
        """Render and save HTML report to target destination."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        html_content = self.render_html(data)
        out.write_text(html_content, encoding="utf-8")
        return out
