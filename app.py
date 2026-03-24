import gradio as gr
import anthropic
import pandas as pd
import os
import io
from datetime import date
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable, Image,
)
from reportlab.lib.enums import TA_CENTER
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CARBON_FACTORS = {
    "Electricite": 0.052, "Gaz": 0.227, "Fioul": 0.324,
    "Reseau de chaleur": 0.110, "Reseau de froid": 0.025, "Bois": 0.030,
}
ENERGY_PRICE = {
    "Electricite": 0.20, "Gaz": 0.08, "Fioul": 0.12,
    "Reseau de chaleur": 0.09, "Reseau de froid": 0.06, "Bois": 0.04,
}
BENCHMARKS = {
    "Bureaux":                      {"eui_best": 33.0,  "eui_median": 99.4,  "eui_p75": 150.1},
    "Enseignement Primaire":        {"eui_best": 51.9,  "eui_median": 106.3, "eui_p75": 142.3},
    "Enseignement Secondaire":      {"eui_best": 58.3,  "eui_median": 100.4, "eui_p75": 126.5},
    "Enseignement Superieur":       {"eui_best": 49.0,  "eui_median": 107.5, "eui_p75": 149.8},
    "Sante - Centres Hospitaliers": {"eui_best": 101.4, "eui_median": 212.9, "eui_p75": 286.9},
    "Hotellerie":                   {"eui_best": 90.2,  "eui_median": 166.1, "eui_p75": 222.8},
    "Commerce de gros":             {"eui_best": 17.4,  "eui_median": 60.1,  "eui_p75": 111.0},
    "Sports":                       {"eui_best": 26.6,  "eui_median": 98.1,  "eui_p75": 158.8},
    "Logistique":                   {"eui_best": 9.1,   "eui_median": 49.9,  "eui_p75": 109.2},
}
AXIOM_PROMPT = (
    "You are AXIOM, an automated EU energy audit AI. "
    "Be professional and concise. Always quantify in kWh, EUR, kgCO2e. "
    "Reference ISO 50002 and EU EED 2023/1791. "
    "Use plain text only, no special characters or markdown."
)


def clean(text):
    return text.encode("ascii", "ignore").decode("ascii").strip()


def make_chart(actual_eui, b, activity):
    labels = ["Best Practice", "Sector Median", "Sector P75", "This Building"]
    values = [b["eui_best"], b["eui_median"], b["eui_p75"], actual_eui]
    bar_colors = [
        "#2ECC71", "#F39C12", "#E74C3C",
        "#0066CC" if actual_eui <= b["eui_median"] else "#C0392B",
    ]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    bars = ax.barh(labels, values, color=bar_colors, height=0.5, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(
            val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
            str(round(val, 1)), va="center", ha="left",
            fontsize=9, fontweight="bold", color="#333333",
        )
    ax.set_xlabel("EUI (kWh/m2/year)", fontsize=9, color="#555555")
    ax.set_title(
        "EUI Benchmark Comparison - " + activity,
        fontsize=10, fontweight="bold", color="#1A1A2E", pad=10,
    )
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.set_xlim(0, max(values) * 1.18)
    legend_patches = [
        mpatches.Patch(color="#2ECC71", label="Best Practice (Top 10%)"),
        mpatches.Patch(color="#F39C12", label="Sector Median"),
        mpatches.Patch(color="#E74C3C", label="Sector P75"),
        mpatches.Patch(
            color="#C0392B" if actual_eui > b["eui_median"] else "#0066CC",
            label="This Building (Poor)" if actual_eui > b["eui_median"] else "This Building",
        ),
    ]
    ax.legend(handles=legend_patches, fontsize=7.5, loc="lower right",
              framealpha=0.7, edgecolor="#CCCCCC")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def build_pdf(benchmark_result, ecm_list, narrative, chart_buf,
              path="/tmp/AXIOM_Report.pdf"):
    BLUE = colors.HexColor("#0066CC")
    DARK = colors.HexColor("#1A1A2E")
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2.5 * cm, bottomMargin=2 * cm,
    )
    s_title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=20,
                             textColor=DARK, alignment=TA_CENTER, spaceAfter=6)
    s_sub   = ParagraphStyle("s", fontName="Helvetica", fontSize=10,
                             textColor=BLUE, alignment=TA_CENTER, spaceAfter=4)
    s_meta  = ParagraphStyle("m", fontName="Helvetica", fontSize=7.5,
                             textColor=colors.grey, alignment=TA_CENTER, spaceAfter=0)
    s_h2    = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12,
                             textColor=BLUE, spaceBefore=14, spaceAfter=6)
    s_h3    = ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=10,
                             textColor=DARK, spaceBefore=8, spaceAfter=4)
    s_body  = ParagraphStyle("b", fontName="Helvetica", fontSize=9,
                             leading=14, spaceAfter=4)
    s_bullet = ParagraphStyle("bl", fontName="Helvetica", fontSize=9,
                              leading=14, spaceAfter=3, leftIndent=14)
    story = []
    story.append(Paragraph("AXIOM Energy Audit Platform", s_title))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("Automated Energy Audit Report - ASHRAE Level I", s_sub))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(
        "Generated: " + date.today().strftime("%d %B %Y") +
        " | Standard: ISO 50002:2014 | Regulation: EU EED 2023/1791", s_meta,
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=10))

    # Section 1 — Building Profile
    story.append(Paragraph("1. Building Profile", s_h2))
    tier_clean = clean(benchmark_result["performance_tier"])
    profile = [
        ["Parameter",         "Value"],
        ["Activity",          benchmark_result["activity"]],
        ["Floor Area",        str(benchmark_result["floor_area_m2"]) + " m2"],
        ["Energy Carrier",    benchmark_result["energy_carrier"]],
        ["Actual EUI",        str(round(benchmark_result["actual_eui"], 1)) + " kWh/m2/year"],
        ["Sector Median EUI", str(benchmark_result["benchmark_median"]) + " kWh/m2/year"],
        ["Best Practice EUI", str(benchmark_result["benchmark_best"]) + " kWh/m2/year"],
        ["Performance Tier",  tier_clean],
        ["Savings Potential",
         str(benchmark_result["savings_potential_kwh"]) + " kWh/yr | EUR " +
         str(benchmark_result["savings_potential_eur_year"]) + "/yr | " +
         str(benchmark_result["savings_potential_co2_kgCO2e"]) + " kgCO2e/yr"],
    ]
    t1 = Table(profile, colWidths=[7 * cm, 10 * cm])
    t1.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",       (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4FF")]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
    ]))
    story.append(t1)

    # Section 1b — EUI Chart
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("1b. EUI Benchmark Chart", s_h2))
    story.append(Image(chart_buf, width=15 * cm, height=6.5 * cm))

    # Section 2 — ECMs
    story.append(Paragraph("2. Energy Conservation Measures", s_h2))
    ecm_data = [["ECM", "kWh/yr", "EUR/yr", "kgCO2e/yr", "CapEx EUR", "Payback", "Priority"]]
    for e in ecm_list:
        ecm_data.append([e["ECM"], e["kWh/yr"], e["EUR/yr"],
                         e["kgCO2e/yr"], e["CapEx EUR"], e["Payback"], e["Priority"]])
    total_kwh = sum(int(e["kWh/yr"].replace(",", "")) for e in ecm_list)
    total_eur = sum(int(e["EUR/yr"].replace(",", "")) for e in ecm_list)
    total_co2 = sum(int(e["kgCO2e/yr"].replace(",", "")) for e in ecm_list)
    ecm_data.append(["TOTAL", str(total_kwh), str(total_eur), str(total_co2), "", "", ""])
    t2 = Table(ecm_data, colWidths=[5.5*cm, 2*cm, 2*cm, 2.3*cm, 2.3*cm, 1.8*cm, 1.5*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0),  (-1, 0),  BLUE),
        ("TEXTCOLOR",      (0, 0),  (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0),  (-1, 0),  "Helvetica-Bold"),
        ("FONTNAME",       (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND",     (0, -1), (-1, -1), colors.HexColor("#E8F5E9")),
        ("FONTSIZE",       (0, 0),  (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1),  (-1, -2), [colors.white, colors.HexColor("#F0F4FF")]),
        ("GRID",           (0, 0),  (-1, -1), 0.5, colors.lightgrey),
        ("ALIGN",          (1, 0),  (-1, -1), "RIGHT"),
        ("LEFTPADDING",    (0, 0),  (-1, -1), 5),
        ("TOPPADDING",     (0, 0),  (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0),  (-1, -1), 4),
    ]))
    story.append(t2)

    # Section 3 — Executive Summary
    story.append(Paragraph("3. Executive Summary", s_h2))
    for line in narrative.split("\n"):
        line = clean(line)
        if not line:
            story.append(Spacer(1, 4))
        elif line.isupper() and len(line) > 4:
            story.append(Paragraph(line.title(), s_h3))
        elif line.startswith("- "):
            story.append(Paragraph("- " + line[2:], s_bullet))
        else:
            story.append(Paragraph(line, s_body))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Paragraph(
        "AXIOM Automated Energy Audit Platform | Powered by Claude AI "
        "| ISO 50002:2014 | EU EED 2023/1791",
        s_meta,
    ))
    doc.build(story)
    return path


def run_audit(activity, floor_area, consumption, carrier):
    b     = BENCHMARKS[activity]
    actual = consumption / floor_area
    cf    = CARBON_FACTORS[carrier]
    price = ENERGY_PRICE[carrier]
    if actual <= b["eui_best"]:
        tier = "Best Practice (Top 10%)"
    elif actual <= b["eui_median"]:
        tier = "Average"
    elif actual <= b["eui_p75"]:
        tier = "Below Average"
    else:
        tier = "Poor (Bottom 25%)"
    gap     = max(0, actual - b["eui_best"])
    sav_kwh = round(gap * floor_area)
    sav_eur = round(sav_kwh * price)
    sav_co2 = round(sav_kwh * cf)
    benchmark_result = {
        "activity": activity, "floor_area_m2": floor_area,
        "actual_eui": actual, "benchmark_best": b["eui_best"],
        "benchmark_median": b["eui_median"], "benchmark_p75": b["eui_p75"],
        "performance_tier": tier, "energy_carrier": carrier,
        "savings_potential_kwh": sav_kwh,
        "savings_potential_eur_year": sav_eur,
        "savings_potential_co2_kgCO2e": sav_co2,
    }
    ecms = []

    def add(name, share, pct, cost_m2, priority):
        kwh  = actual * floor_area * share * pct
        eur  = kwh * price
        cost = floor_area * cost_m2
        ecms.append({
            "ECM": name,
            "kWh/yr": str(round(kwh)), "EUR/yr": str(round(eur)),
            "kgCO2e/yr": str(round(kwh * cf)), "CapEx EUR": str(round(cost)),
            "Payback": str(round(cost / eur, 1)) + " yrs", "Priority": priority,
            "_kwh": kwh, "_eur": eur, "_co2": kwh * cf,
        })

    add("BMS Optimisation",        1.00, 0.15,  8, "High")
    add("LED Lighting + Controls", 0.20, 0.60, 15, "High")
    if actual > b["eui_median"]:
        add("HVAC Heat Pump Upgrade", 1.00, 0.25, 80, "Medium")
    if actual > b["eui_median"] * 1.3:
        add("Building Envelope",      1.00, 0.10, 120, "Low")
    ecms.sort(key=lambda x: float(x["Payback"].split()[0]))
    total_kwh = round(sum(e["_kwh"] for e in ecms))
    total_eur = round(sum(e["_eur"] for e in ecms))
    total_co2 = round(sum(e["_co2"] for e in ecms))
    benchmark_md = "## Benchmark Results\n"
    benchmark_md += "| Metric | Value |\n|--------|-------|\n"
    benchmark_md += "| **Actual EUI** | " + str(round(actual, 1)) + " kWh/m2/yr |\n"
    benchmark_md += "| **Sector Median** | " + str(b["eui_median"]) + " kWh/m2/yr |\n"
    benchmark_md += "| **Best Practice** | " + str(b["eui_best"]) + " kWh/m2/yr |\n"
    benchmark_md += "| **Performance** | " + tier + " |\n"
    benchmark_md += (
        "| **Savings Potential** | " + str(sav_kwh) + " kWh/yr / EUR "
        + str(sav_eur) + "/yr / " + str(sav_co2) + " kgCO2e/yr |\n"
    )
    df = pd.DataFrame([{k: v for k, v in e.items() if not k.startswith("_")} for e in ecms])
    ecm_md = "## ECM Priority List\n" + df.to_markdown(index=False)
    ecm_md += (
        "\n\n**Total:** " + str(total_kwh) + " kWh/yr / EUR "
        + str(total_eur) + "/yr / " + str(total_co2) + " kgCO2e/yr"
    )
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    ecm_text = "\n".join([
        "- " + e["ECM"] + ": " + e["kWh/yr"] + " kWh/yr, EUR "
        + e["EUR/yr"] + "/yr, " + e["Payback"] + " payback"
        for e in ecms
    ])
    prompt = (
        "Write a 200-word professional energy audit executive summary. "
        "Plain ASCII text only. No markdown, no special characters, no bullet symbols. "
        "Use short ALL CAPS lines as section headers (e.g. CURRENT PERFORMANCE, RECOMMENDATIONS). "
        "Building: " + activity + ", " + str(floor_area) + " m2, EUI "
        + str(round(actual, 1)) + " kWh/m2/yr, performance: " + tier + "\n"
        "ECMs:\n" + ecm_text + "\n"
        "Total savings: " + str(total_kwh) + " kWh/yr, EUR "
        + str(total_eur) + "/yr, " + str(total_co2) + " kgCO2e/yr"
    )
    msg = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=500,
        system=AXIOM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    narrative = msg.content[0].text
    chart_buf = make_chart(actual, b, activity)
    pdf_path  = build_pdf(benchmark_result, ecms, narrative, chart_buf)
    return benchmark_md, ecm_md, narrative, pdf_path


with gr.Blocks(title="AXIOM Energy Audit") as demo:
    gr.Markdown("""
# AXIOM Energy Audit Platform
### Automated EU-compliant energy audits powered by Claude AI
*Benchmarked against 1M+ French tertiary buildings (ADEME open data)*
---
""")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Building Profile")
            activity    = gr.Dropdown(list(BENCHMARKS.keys()), value="Bureaux",
                                      label="Activity Type")
            floor_area  = gr.Slider(500, 50000, value=5000,
                                    step=500, label="Floor Area (m2)")
            consumption = gr.Slider(50000, 5000000, value=800000,
                                    step=10000, label="Annual Consumption (kWh)")
            carrier     = gr.Dropdown(list(CARBON_FACTORS.keys()), value="Electricite",
                                      label="Energy Carrier")
            btn         = gr.Button("Run AXIOM Audit", variant="primary", size="lg")
        with gr.Column(scale=2):
            benchmark_out = gr.Markdown()
            ecm_out       = gr.Markdown()
    gr.Markdown("### Executive Summary")
    narrative_out = gr.Textbox(label="", lines=8)
    gr.Markdown("### Download Report")
    pdf_out = gr.File(label="AXIOM_Report.pdf", file_types=[".pdf"])
    gr.Markdown("---\n*AXIOM | ISO 50002:2014 | EU EED 2023/1791 | Powered by Claude AI*")
    btn.click(
        fn=run_audit,
        inputs=[activity, floor_area, consumption, carrier],
        outputs=[benchmark_out, ecm_out, narrative_out, pdf_out],
    )

demo.launch(theme=gr.themes.Soft())
