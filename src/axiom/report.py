"""AXIOM PDF Report Generator."""
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER
from datetime import date

BLUE = colors.HexColor("#0066CC")
DARK = colors.HexColor("#1A1A2E")

def md_to_rl(text):
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", text)
    return re.sub(r"[^\x00-\x7F]", "", text).strip()

def generate_pdf_report(benchmark_result, ecm_list, narrative,
                         output_path="AXIOM_Audit_Report.pdf"):
    from axiom.report_builder import build_story
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2.5*cm, bottomMargin=2*cm)
    story = build_story(benchmark_result, ecm_list, narrative)
    doc.build(story)
    return output_path
