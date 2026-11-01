"""Report generator for Siberian Bank Collateral Inspection Act (Приложение 1).

Renders official Jinja2 HTML reports and handles export to file and PDF.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Optional

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    _HAS_JINJA2 = True
except ImportError:
    _HAS_JINJA2 = False

from ..models import ActReportData

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


class ActGenerator:
    """Generates official inspection act documents in HTML and vector PDF formats."""

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
            f"<p>Номер акта: {data.act_number} | Дата: {data.act_date} | Система мониторинга залогов Сбербанк | "
            f"{data.branch_name}</p>"
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

    def render_pdf(self, data: ActReportData) -> bytes:
        """Render ActReportData to official vector PDF format."""
        html_content = self.render_html(data)
        return self.html_to_pdf(html_content, data=data)

    def save_pdf(self, data: ActReportData, output_path: Path | str) -> Path:
        """Render and save PDF report to target destination."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pdf_bytes = self.render_pdf(data)
        out.write_bytes(pdf_bytes)
        return out

    @classmethod
    def html_to_pdf(cls, html_content: str, data: Optional[ActReportData] = None) -> bytes:
        """Convert rendered HTML string into a standalone PDF document."""
        # 1. Primary engine: Headless Google Chrome / Chromium
        chrome_candidates = [
            "/usr/bin/google-chrome",
            "google-chrome",
            "/usr/bin/chromium",
            "chromium",
            "/usr/bin/chromium-browser",
            "chromium-browser",
        ]
        found_bin = None
        for cand in chrome_candidates:
            if cand.startswith("/") and Path(cand).exists():
                found_bin = cand
                break
            elif shutil.which(cand):
                found_bin = cand
                break

        if found_bin:
            with tempfile.NamedTemporaryFile(suffix=".html", mode="w", encoding="utf-8", delete=False) as h_tmp:
                h_tmp.write(html_content)
                h_path = h_tmp.name

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as p_tmp:
                p_path = p_tmp.name

            try:
                cmd = [
                    found_bin,
                    "--headless=new",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--run-all-compositor-stages-before-draw",
                    "--print-to-pdf-no-header",
                    f"--print-to-pdf={p_path}",
                    h_path,
                ]
                proc = subprocess.run(cmd, capture_output=True, timeout=25)
                if proc.returncode == 0 and Path(p_path).exists():
                    pdf_bytes = Path(p_path).read_bytes()
                    if pdf_bytes.startswith(b"%PDF"):
                        return pdf_bytes
            except Exception:
                pass
            finally:
                Path(h_path).unlink(missing_ok=True)
                Path(p_path).unlink(missing_ok=True)

        # 2. Resilient fallback: ReportLab canvas
        return cls._render_fallback_pdf(data)

    @classmethod
    def _render_fallback_pdf(cls, data: Optional[ActReportData] = None) -> bytes:
        """Fallback lightweight PDF generation via ReportLab when headless browser is absent."""
        from io import BytesIO
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas

            buf = BytesIO()
            c = canvas.Canvas(buf, pagesize=A4)
            c.setFont("Helvetica-Bold", 14)
            act_num = data.act_number if data else "АКТ-СИБ-2026-0001"
            c.drawString(50, 800, f"АКТ ПРОВЕРКИ ИМУЩЕСТВА № {act_num}")
            c.setFont("Helvetica", 10)
            if data:
                c.drawString(50, 780, f"Дата составления: {data.act_date}")
                c.drawString(50, 760, f"Кадастровый номер: {data.credit.cadastral_number}")
                c.drawString(50, 740, f"Физическая готовность: {data.inspection.physical_readiness_pct:.1f}%")
                c.drawString(50, 720, f"Рекомендованная стратегия: {data.recommendation.strategy.value}")
                c.drawString(50, 700, f"Материал стен: {data.inspection.detected_wall_material.value}")
            c.save()
            return buf.getvalue()
        except Exception:
            # Minimal hardcoded PDF 1.4 header and trailer if no libraries are available
            return (
                b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
                b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
                b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] >>\nendobj\n"
                b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n"
                b"0000000115 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n185\n%%EOF"
            )
