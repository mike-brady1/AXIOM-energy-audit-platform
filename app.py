
import gradio as gr
import anthropic
import pandas as pd
import os
import io
import zipfile
from io import StringIO
from datetime import date
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable, Image,
                                 KeepTogether)
from reportlab.lib.enums import TA_CENTER
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pdfplumber
import requests

CARBON_FACTORS = {
    "Electricite": 0.052, "Gaz": 0.227, "Fioul": 0.324,
    "Reseau de chaleur": 0.110, "Reseau de froid": 0.025, "Bois": 0.030,
}
ENERGY_PRICE = {
    "Electricite": 0.20, "Gaz": 0.08, "Fioul": 0.12,
    "Reseau de chaleur": 0.09, "Reseau de froid": 0.06, "Bois": 0.04,
}
BENCHMARKS = {
    "Bureaux":                       {"eui_best": 33.0,  "eui_median": 99.4,  "eui_p75": 150.1},
    "Enseignement Primaire":         {"eui_best": 51.9,  "eui_median": 106.3, "eui_p75": 142.3},
    "Enseignement Secondaire":       {"eui_best": 58.3,  "eui_median": 100.4, "eui_p75": 126.5},
    "Enseignement Superieur":        {"eui_best": 49.0,  "eui_median": 107.5, "eui_p75": 149.8},
    "Sante - Centres Hospitaliers":  {"eui_best": 101.4, "eui_median": 212.9, "eui_p75": 286.9},
    "Hotellerie":                    {"eui_best": 90.2,  "eui_median": 166.1, "eui_p75": 222.8},
    "Commerce de gros":              {"eui_best": 17.4,  "eui_median": 60.1,  "eui_p75": 111.0},
    "Sports":                        {"eui_best": 26.6,  "eui_median": 98.1,  "eui_p75": 158.8},
    "Logistique":                    {"eui_best": 9.1,   "eui_median": 49.9,  "eui_p75": 109.2},
}
AXIOM_PROMPT = "You are AXIOM, an automated EU energy audit AI. Be professional and concise. Always quantify in kWh, EUR, kgCO2e. Reference ISO 50002 and EU EED 2023/1791. Use plain text only, no special characters or markdown."
VALID_ACTIVITIES = list(BENCHMARKS.keys())
VALID_CARRIERS   = list(CARBON_FACTORS.keys())

BATCH_TEMPLATE = """building_name,activity,floor_area_m2,consumption_kwh,energy_carrier
Tour Montparnasse,Bureaux,15000,2100000,Electricite
Lycee Victor Hugo,Enseignement Secondaire,8000,720000,Gaz
Hotel Mercure Lyon,Hotellerie,6000,950000,Electricite
Entrepot Rungis,Logistique,25000,800000,Fioul
Clinique Saint-Louis,Sante - Centres Hospitaliers,12000,3200000,Electricite
"""

TIER_COLOR = {
    "Best Practice (Top 10%)": "#27AE60",
    "Average":                 "#F39C12",
    "Below Average":           "#E67E22",
    "Poor (Bottom 25%)":       "#E74C3C",
}


# ─────────────────────────────────────────────────────────────────────────────
# AUTH + BRANDING
# ─────────────────────────────────────────────────────────────────────────────
import hashlib, hmac

def _check_password(username: str, password: str) -> bool:
    """Validate against HF Space secrets AXIOM_USER / AXIOM_PASS.
    Falls back to open access if secrets are not set (dev mode)."""
    expected_user = os.environ.get("AXIOM_USER", "")
    expected_pass = os.environ.get("AXIOM_PASS", "")
    if not expected_user and not expected_pass:
        return True   # dev mode — no secrets set
    user_ok = hmac.compare_digest(username.strip(), expected_user.strip())
    pass_ok = hmac.compare_digest(
        hashlib.sha256(password.encode()).hexdigest(),
        hashlib.sha256(expected_pass.encode()).hexdigest()
    )
    return user_ok and pass_ok

# Branding defaults (override via HF Secrets)
ORG_NAME    = os.environ.get("AXIOM_ORG_NAME",  "AXIOM Energy Audit Platform")
ORG_TAGLINE = os.environ.get("AXIOM_TAGLINE",   "Powered by Claude AI | ISO 50002:2014 | EU EED 2023/1791")
BRAND_COLOR = os.environ.get("AXIOM_COLOR",     "#1A3C5E")   # deep navy default

# ── Reference HDD (base 18°C) for major French cities ───────────────────────
HDD_REFERENCE = {
    "Paris": 2500, "Lyon": 2400, "Marseille": 1600, "Bordeaux": 2100,
    "Lille": 2900, "Strasbourg": 2800, "Toulouse": 1900, "Nantes": 2200,
    "Nice": 1200, "Rennes": 2300, "Grenoble": 2600, "Montpellier": 1700,
}
HDD_STANDARD = 2500  # Paris baseline for normalisation

def fetch_hdd(city: str, year: int = 2024) -> dict:
    """Fetch annual HDD (base 18C) from Open-Meteo for a given city."""
    coords = {
        "Paris":      (48.85, 2.35),  "Lyon":       (45.75, 4.85),
        "Marseille":  (43.30, 5.37),  "Bordeaux":   (44.84, -0.58),
        "Lille":      (50.63, 3.06),  "Strasbourg": (48.57, 7.75),
        "Toulouse":   (43.60, 1.44),  "Nantes":     (47.22, -1.55),
        "Nice":       (43.71, 7.26),  "Rennes":     (48.11, -1.68),
        "Grenoble":   (45.19, 5.72),  "Montpellier":(43.61, 3.87),
    }
    lat, lon = coords.get(city, (48.85, 2.35))  # default Paris
    try:
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={year}-01-01&end_date={year}-12-31"
            f"&daily=temperature_2m_mean&timezone=Europe%2FParis"
        )
        r = requests.get(url, timeout=10)
        temps = r.json()["daily"]["temperature_2m_mean"]
        hdd   = sum(max(0, 18 - t) for t in temps if t is not None)
        return {"city": city, "year": year, "hdd_actual": round(hdd),
                "hdd_standard": HDD_STANDARD, "source": "Open-Meteo"}
    except Exception:
        hdd_ref = HDD_REFERENCE.get(city, HDD_STANDARD)
        return {"city": city, "year": year, "hdd_actual": hdd_ref,
                "hdd_standard": HDD_STANDARD, "source": "Reference table"}

def weather_normalise(consumption_kwh: float, hdd_actual: int,
                      hdd_standard: int, heating_fraction: float = 0.6) -> dict:
    """
    Normalise consumption to standard HDD year.
    heating_fraction: share of consumption that is weather-sensitive (default 60%).
    """
    weather_sensitive = consumption_kwh * heating_fraction
    base_load         = consumption_kwh * (1 - heating_fraction)
    if hdd_actual == 0:
        norm = consumption_kwh
    else:
        norm = base_load + weather_sensitive * (hdd_standard / hdd_actual)
    correction_pct = round((norm - consumption_kwh) / consumption_kwh * 100, 1)
    return {
        "normalised_kwh":   round(norm),
        "correction_pct":   correction_pct,
        "hdd_actual":       hdd_actual,
        "hdd_standard":     hdd_standard,
        "weather_note":     (
            f"Warmer than standard year (HDD {hdd_actual} vs {hdd_standard}): "
            f"normalised consumption +{correction_pct}% — building may be "
            f"more efficient than raw EUI suggests."
        ) if correction_pct > 0 else (
            f"Colder than standard year (HDD {hdd_actual} vs {hdd_standard}): "
            f"normalised consumption {correction_pct}% — raw EUI overstates efficiency."
        )
    }


# EU ETS shadow carbon price (EUR/tCO2e) — update annually
EU_ETS_PRICE = 65.0



# v22c-forced-restart
# ── DPE thresholds (kWh EP/m2/yr) — Decret 2021-872 tertiary ────────────────
# Primary energy = final energy × primary energy factor
# EP factors: Elec=2.3, Gaz=1.0, Fioul=1.0, RCU=1.0, RFR=1.0, Bois=1.0
EP_FACTOR = {
    "Electricite": 2.3, "Gaz": 1.0, "Fioul": 1.0,
    "Reseau de chaleur": 1.0, "Reseau de froid": 1.0, "Bois": 1.0,
}
# DPE energy class thresholds (primary energy kWh EP/m2/yr)
DPE_ENERGY = [
    ("A", 0,   70,  "#319834"),
    ("B", 70,  110, "#33CC33"),
    ("C", 110, 180, "#CDFF33"),
    ("D", 180, 250, "#FFFF00"),
    ("E", 250, 330, "#FFCC00"),
    ("F", 330, 420, "#FF6600"),
    ("G", 420, 9999,"#FF0000"),
]
# DPE GHG class thresholds (kgCO2e/m2/yr)
DPE_GHG = [
    ("A", 0,   6,   "#319834"),
    ("B", 6,   11,  "#33CC33"),
    ("C", 11,  30,  "#CDFF33"),
    ("D", 30,  50,  "#FFFF00"),
    ("E", 50,  70,  "#FFCC00"),
    ("F", 70,  100, "#FF6600"),
    ("G", 100, 9999,"#FF0000"),
]


# ── Décret Tertiaire trajectory ──────────────────────────────────────────────
TERTIAIRE_TARGETS = {2030: 0.40, 2040: 0.50, 2050: 0.60}   # % reduction vs baseline

def make_tertiaire_chart(baseline_eui, activity, floor_area):
    b        = BENCHMARKS[activity]
    years    = [2024, 2030, 2040, 2050]
    pcts     = [0, 40, 50, 60]
    targets  = [baseline_eui * (1 - p/100) for p in pcts]
    best_eui = b["eui_best"]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    ax.axhline(baseline_eui, color="#E74C3C", linestyle="--",
               linewidth=1.5, label="Baseline EUI", zorder=2)
    ax.plot(years, targets, color="#0066CC", linewidth=2.5,
            marker="o", markersize=7, label="Decret Tertiaire Target", zorder=4)
    ax.fill_between(years, targets, [baseline_eui]*len(years),
                    alpha=0.08, color="#E74C3C", label="Required reduction zone")

    for yr, eui, pct in zip(years, targets, pcts):
        if pct > 0:
            label = "-" + str(pct) + "% | " + str(round(eui, 1)) + " kWh/m2"
            ax.annotate(label,
                xy=(yr, eui),
                xytext=(yr, eui - baseline_eui * 0.09),
                ha="center", fontsize=8, fontweight="bold", color="#0066CC",
                arrowprops=dict(arrowstyle="-", color="#0066CC", lw=1))

    ax.axhline(best_eui, color="#27AE60", linestyle=":",
               linewidth=1.5,
               label="Best Practice (" + str(best_eui) + " kWh/m2/yr)", zorder=2)
    ax.axhline(0, color="#CCCCCC", linewidth=0.8)
    ax.axvline(2026, color="#F39C12", linestyle="--", linewidth=1.2, alpha=0.7, zorder=3)
    ax.text(2026.1, baseline_eui * 0.95, "OPERAT 30 Sep 2026",
            fontsize=7.5, color="#F39C12", va="top")

    ax.set_xlim(2023, 2052)
    ax.set_ylim(0, baseline_eui * 1.15)
    ax.set_xticks(years)
    ax.set_xlabel("Year", fontsize=9, color="#555555")
    ax.set_ylabel("EUI (kWh/m2/yr)", fontsize=9, color="#555555")
    ax.set_title("Decret Tertiaire 2019-771 — Reduction Trajectory | " + activity,
                 fontsize=10, fontweight="bold", color="#1A1A2E", pad=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85, edgecolor="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.tick_params(labelsize=8)

    kpi = ("Floor area: " + str(floor_area) + " m2  |  "
           "2030: " + str(round(targets[1],1)) + "  |  "
           "2040: " + str(round(targets[2],1)) + "  |  "
           "2050: " + str(round(targets[3],1)) + " kWh/m2/yr")
    ax.text(0.01, 0.97, kpi, transform=ax.transAxes,
            fontsize=7.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.9))

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf, targets

def get_dpe(eui_final, carrier, carbon_intensity_m2):
    """Return (energy_class, ghg_class, ep_kWh_m2, hex_energy, hex_ghg)."""
    ep = eui_final * EP_FACTOR.get(carrier, 1.0)
    e_class = "G"; e_color = "#FF0000"
    for label, lo, hi, color in DPE_ENERGY:
        if lo <= ep < hi:
            e_class = label; e_color = color; break
    g_class = "G"; g_color = "#FF0000"
    for label, lo, hi, color in DPE_GHG:
        if lo <= carbon_intensity_m2 < hi:
            g_class = label; g_color = color; break
    # Worst of the two classes is the final DPE label
    order = ["A","B","C","D","E","F","G"]
    final = e_class if order.index(e_class) >= order.index(g_class) else g_class
    final_color = e_color if final == e_class else g_color
    return {
        "energy_class": e_class, "energy_color": e_color,
        "ghg_class": g_class,    "ghg_color": g_color,
        "final_class": final,    "final_color": final_color,
        "ep_kWh_m2": round(ep, 1),
        "carbon_m2": round(carbon_intensity_m2, 1),
    }

def make_dpe_badge(dpe):
    """Official-style DPE arrow bars. GES in separate subplot — never overlaps."""
    order = ["A", "B", "C", "D", "E", "F", "G"]
    DPE_COLORS = {
        "A": "#009900", "B": "#55BB00", "C": "#AACC00",
        "D": "#FFCC00", "E": "#FF9900", "F": "#FF5500", "G": "#DD0000",
    }
    BAR_W  = {lbl: 2.8 + i * 0.65 for i, lbl in enumerate(order)}
    ARROW  = 0.32
    BAR_H  = 0.55
    GAP    = 0.90
    N      = len(order)
    MAX_W  = BAR_W["G"] + ARROW

    fig, (ax, ax_ges) = plt.subplots(
        2, 1, figsize=(5.5, N * GAP * 0.65 + 0.9),
        gridspec_kw={"height_ratios": [N * GAP, 0.7], "hspace": 0.05}
    )
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(-0.3, MAX_W + 1.5)
    ax.set_ylim(-BAR_H, N * GAP)
    ax.axis("off")
    ax.set_title("DPE — Diagnostic de Performance Energetique",
                 fontsize=9.5, fontweight="bold", color="#1A1A2E", pad=8)

    from matplotlib.patches import Polygon as MplPolygon
    for i, lbl in enumerate(order):
        y      = (N - 1 - i) * GAP
        w      = BAR_W[lbl]
        c      = DPE_COLORS[lbl]
        active = (lbl == dpe["final_class"])
        alpha  = 1.0 if active else 0.28
        tip_x  = w + ARROW
        half   = BAR_H / 2
        pts    = [(0, y-half), (w, y-half), (tip_x, y), (w, y+half), (0, y+half)]
        poly = MplPolygon(pts, closed=True, facecolor=c, alpha=alpha,
                          edgecolor="white", linewidth=1.0, zorder=3)
        ax.add_patch(poly)
        ax.text(MAX_W + 0.3, y, lbl,
                va="center", ha="left",
                fontsize=11 if active else 8.5,
                fontweight="bold" if active else "normal",
                color="#1A1A2E", alpha=1.0 if active else 0.5, zorder=4)
        if active:
            ax.text(w - 0.12, y,
                    str(dpe["ep_kWh_m2"]) + " kWh EP/m2/yr",
                    va="center", ha="right",
                    fontsize=8, fontweight="bold", color="white", zorder=5)
            border = MplPolygon(pts, closed=True, facecolor="none",
                                edgecolor="#1A1A2E", linewidth=2.2, zorder=6)
            ax.add_patch(border)

    ax_ges.set_facecolor("#F8F8F8")
    ax_ges.axis("off")
    ghg_c = DPE_COLORS.get(dpe["ghg_class"], "#888888")
    ax_ges.text(0.02, 0.5,
                "GES Class " + dpe["ghg_class"] +
                "   |   " + str(dpe["carbon_m2"]) + " kgCO2e/m2/yr",
                va="center", ha="left", transform=ax_ges.transAxes,
                fontsize=8.5, fontweight="bold", color=ghg_c)

    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf

def clean(text):
    return text.encode("ascii", "ignore").decode("ascii").strip()

def read_input_file(uploaded_file, client):
    path = uploaded_file.name
    ext  = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path), None
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path, engine="openpyxl"), None
    elif ext == ".ods":
        return pd.read_excel(path, engine="odf"), None
    elif ext == ".pdf":
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        if not text.strip():
            return None, "Could not extract text from PDF."
        msg = client.messages.create(
            model="claude-sonnet-4-5", max_tokens=1500,
            system=(
                "Extract building energy data. Return ONLY valid CSV with no extra text. "
                "Required columns: building_name, activity, floor_area_m2, consumption_kwh, energy_carrier. "
                "Valid activities: " + ", ".join(VALID_ACTIVITIES) + ". "
                "Valid carriers: " + ", ".join(VALID_CARRIERS) + ". "
                "Return ONLY CSV rows — no explanation, no markdown, no code block."
            ),
            messages=[{"role": "user", "content": "Extract all buildings:\n\n" + text[:6000]}]
        )
        return pd.read_csv(StringIO(msg.content[0].text.strip())), None
    else:
        return None, "Unsupported file type: " + ext + ". Use CSV, XLSX, ODS or PDF."

def check_compliance(floor_area, consumption):
    flags = []
    flags.append({"regulation": "EU EED 2023/1791 Art.11", "status": "MANDATORY",
                  "obligation": "Energy audit every 4 years for non-SME enterprises",
                  "deadline": "Within 4 years of last audit", "color": "RED"})
    if consumption >= 23600000:
        flags.append({"regulation": "EU EED 2023/1791 Art.8", "status": "MANDATORY",
                      "obligation": "Implement ISO 50001 Energy Management System",
                      "deadline": "By 11 October 2027", "color": "RED"})
    if floor_area >= 1000:
        flags.append({"regulation": "Decret Tertiaire (FR) 2019-771", "status": "MANDATORY",
                      "obligation": "-40% by 2030 / -50% by 2040 / -60% by 2050 vs 2010",
                      "deadline": "OPERAT reporting 30 Sep 2026", "color": "RED"})
    flags.append({"regulation": "CSRD / ESRS E1", "status": "LIKELY APPLICABLE",
                  "obligation": "Report Scope 1/2/3 GHG + energy in sustainability report",
                  "deadline": "FY2025 large / FY2026 listed SMEs", "color": "ORANGE"})
    flags.append({"regulation": "ISO 50001:2018", "status": "RECOMMENDED",
                  "obligation": "Implement Energy Management System",
                  "deadline": "Best practice — no deadline", "color": "GREEN"})
    flags.append({"regulation": "IPMVP (EVO 10000)", "status": "RECOMMENDED",
                  "obligation": "Apply M+V protocol to verify ECM savings",
                  "deadline": "Within 12 months of ECM commissioning", "color": "GREEN"})
    return flags

def compliance_color(color_str):
    return {"RED": colors.HexColor("#E74C3C"),
            "ORANGE": colors.HexColor("#F39C12"),
            "GREEN":  colors.HexColor("#27AE60")}.get(color_str, colors.grey)

def make_chart(actual_eui, b, activity):
    labels = ["Best Practice", "Sector Median", "Sector P75", "This Building"]
    values = [b["eui_best"], b["eui_median"], b["eui_p75"], actual_eui]
    bar_colors = ["#2ECC71", "#F39C12", "#E74C3C",
                  "#0066CC" if actual_eui <= b["eui_median"] else "#C0392B"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    bars = ax.barh(labels, values, color=bar_colors, height=0.5, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(val + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                str(round(val,1)), va="center", ha="left",
                fontsize=9, fontweight="bold", color="#333333")
    ax.set_xlabel("EUI (kWh/m2/year)", fontsize=9, color="#555555")
    ax.set_title("EUI Benchmark - " + activity, fontsize=10,
                 fontweight="bold", color="#1A1A2E", pad=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(values) * 1.18)
    ax.legend(handles=[
        mpatches.Patch(color="#2ECC71", label="Best Practice"),
        mpatches.Patch(color="#F39C12", label="Sector Median"),
        mpatches.Patch(color="#E74C3C", label="Sector P75"),
        mpatches.Patch(color="#C0392B" if actual_eui > b["eui_median"] else "#0066CC",
                       label="This Building"),
    ], fontsize=7.5, loc="lower right", framealpha=0.7, edgecolor="#CCCCCC")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# ── STEP 19: Portfolio Dashboard ─────────────────────────────────────────────
def make_portfolio_dashboard(results):
    """
    3-panel dashboard:
    Panel A — EUI per building vs sector median (horizontal bars, colour by tier)
    Panel B — Annual savings EUR/yr per building (sorted descending)
    Panel C — Portfolio donut: buildings by performance tier
    """
    names     = [r["Building"] for r in results]
    euis      = [r["EUI"] for r in results]
    tiers     = [r["Performance"] for r in results]
    savings   = [r["Savings EUR/yr"] for r in results]
    activities= [r["Activity"] for r in results]
    medians   = [BENCHMARKS[a]["eui_median"] for a in activities]
    bar_colors= [TIER_COLOR.get(t, "#95A5A6") for t in tiers]

    # Short names for display
    short_names = [n[:22] + "…" if len(n) > 22 else n for n in names]

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle("AXIOM Portfolio Energy Audit Dashboard",
                 fontsize=16, fontweight="bold", color="#1A1A2E", y=0.98)

    # ── Panel A: EUI vs Sector Median ────────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, (1, 3))   # spans left column, both rows
    y   = np.arange(len(names))
    h   = 0.35

    bars_eui = ax1.barh(y + h/2, euis, height=h,
                        color=bar_colors, edgecolor="white", label="Actual EUI", zorder=3)
    bars_med = ax1.barh(y - h/2, medians, height=h,
                        color="#BDC3C7", edgecolor="white", label="Sector Median", zorder=3)

    for bar, val in zip(bars_eui, euis):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2,
                 str(round(val)), va="center", ha="left", fontsize=8, fontweight="bold")
    for bar, val in zip(bars_med, medians):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2,
                 str(round(val)), va="center", ha="left", fontsize=7.5,
                 color="#7F8C8D")

    ax1.set_yticks(y)
    ax1.set_yticklabels(short_names, fontsize=9)
    ax1.set_xlabel("EUI (kWh/m²/year)", fontsize=9)
    ax1.set_title("A — Actual EUI vs Sector Median (coloured by performance tier)",
                  fontsize=10, fontweight="bold", color="#1A1A2E", pad=8)
    ax1.set_facecolor("#FAFAFA")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax1.set_xlim(0, max(euis + medians) * 1.20)

    legend_patches = [mpatches.Patch(color=c, label=t)
                      for t, c in TIER_COLOR.items()]
    legend_patches.append(mpatches.Patch(color="#BDC3C7", label="Sector Median"))
    ax1.legend(handles=legend_patches, fontsize=8, loc="lower right",
               framealpha=0.8, edgecolor="#CCCCCC")

    # ── Panel B: Savings EUR/yr ───────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    sorted_idx  = np.argsort(savings)[::-1]
    s_names     = [short_names[i] for i in sorted_idx]
    s_savings   = [savings[i] for i in sorted_idx]
    s_colors    = [bar_colors[i] for i in sorted_idx]

    bars2 = ax2.barh(s_names, s_savings, color=s_colors, edgecolor="white", height=0.5)
    for bar, val in zip(bars2, s_savings):
        ax2.text(val + max(s_savings)*0.01, bar.get_y() + bar.get_height()/2,
                 "EUR " + f"{val:,}", va="center", ha="left", fontsize=8, fontweight="bold")
    ax2.set_xlabel("EUR savings / year", fontsize=9)
    ax2.set_title("B — Annual Savings Potential (EUR/yr)", fontsize=10,
                  fontweight="bold", color="#1A1A2E", pad=8)
    ax2.set_facecolor("#FAFAFA")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    ax2.set_xlim(0, max(s_savings) * 1.30)

    # ── Panel C: Donut — tier breakdown ──────────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 4)
    tier_counts = {}
    for t in tiers:
        tier_counts[t] = tier_counts.get(t, 0) + 1
    total_bldgs  = sum(tier_counts.values())
    donut_keys   = list(tier_counts.keys())
    donut_labels = [f"{k} ({round(v/total_bldgs*100)}%)" 
                    for k, v in tier_counts.items()]
    donut_sizes  = list(tier_counts.values())
    donut_colors = [TIER_COLOR.get(k, "#95A5A6") for k in donut_keys]

    wedges, texts, autotexts = ax3.pie(
        donut_sizes, labels=None, colors=donut_colors,
        autopct="%1.0f%%", startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
        at.set_color("white")

    ax3.legend(wedges, [l + " (" + str(c) + ")" for l, c in zip(donut_labels, donut_sizes)],
               fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.12),
               framealpha=0.8, edgecolor="#CCCCCC")
    ax3.set_title("C — Portfolio Tier Breakdown", fontsize=10,
                  fontweight="bold", color="#1A1A2E", pad=8)

    # Total savings annotation in donut centre
    total_eur = sum(savings)
    ax3.text(0, 0, "EUR\n" + f"{total_eur:,}\n/yr",
             ha="center", va="center", fontsize=9,
             fontweight="bold", color="#1A1A2E")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#F8F9FA")
    plt.close(fig)
    buf.seek(0)
    return buf

def build_pdf(bname, benchmark_result, ecm_list, narrative,
              chart_buf, compliance_flags, path, dpe=None, dpe_badge_buf=None,
              tertiaire_buf=None, tertiaire_targets=None):
    BLUE = colors.HexColor("#0066CC")
    DARK = colors.HexColor("#1A1A2E")
    doc  = SimpleDocTemplate(path, pagesize=A4,
                              rightMargin=2*cm, leftMargin=2*cm,
                              topMargin=2.5*cm, bottomMargin=2*cm)
    s_title  = ParagraphStyle("t",  fontName="Helvetica-Bold", fontSize=18,
                               textColor=DARK, alignment=TA_CENTER, spaceAfter=6)
    s_sub    = ParagraphStyle("s",  fontName="Helvetica", fontSize=10,
                               textColor=BLUE, alignment=TA_CENTER, spaceAfter=6)
    s_meta   = ParagraphStyle("m",  fontName="Helvetica", fontSize=7.5,
                               textColor=colors.grey, alignment=TA_CENTER, spaceAfter=4)
    s_h2     = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11,
                               textColor=BLUE, spaceBefore=12, spaceAfter=5)
    s_h3     = ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=9,
                               textColor=DARK, spaceBefore=6, spaceAfter=3)
    s_body   = ParagraphStyle("b",  fontName="Helvetica", fontSize=8.5,
                               leading=13, spaceAfter=3)
    s_bullet = ParagraphStyle("bl", fontName="Helvetica", fontSize=8.5,
                               leading=13, spaceAfter=2, leftIndent=12)
    story = []
    story.append(Paragraph(ORG_NAME, s_title))
    story.append(Paragraph(clean(bname), s_sub))
    story.append(Spacer(1, 0.1*cm))
    story.append(Paragraph(
        "Generated: " + date.today().strftime("%d %B %Y") +
        " | ISO 50002:2014 | EU EED 2023/1791", s_meta))
    story.append(Spacer(1, 0.25*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=10))
    story.append(Paragraph("1. Building Profile", s_h2))
    profile = [
        ["Parameter", "Value"],
        ["Building",       clean(bname)],
        ["Activity",       benchmark_result["activity"]],
        ["Floor Area",     str(benchmark_result["floor_area_m2"]) + " m2"],
        ["Energy Carrier", benchmark_result["energy_carrier"]],
        ["Actual EUI",     str(round(benchmark_result["actual_eui"], 1)) + " kWh/m2/yr"],
        ["Sector Median",  str(benchmark_result["benchmark_median"]) + " kWh/m2/yr"],
        ["Best Practice",  str(benchmark_result["benchmark_best"]) + " kWh/m2/yr"],
        ["Performance",    clean(benchmark_result["performance_tier"])],
        ["Weather (HDD actual/std)",
         str(benchmark_result.get("hdd_actual","N/A")) + " / " +
         str(HDD_STANDARD) + "  |  Normalised: " +
         str(benchmark_result.get("normalised_kwh","N/A")) + " kWh"],
        ["Carbon Liability (EU ETS)",
         "EUR " + str(benchmark_result.get("carbon_liability_eur", "N/A")) +
         "/yr  (@EUR " + str(int(EU_ETS_PRICE)) + "/tCO2e)"],
        ["Savings",
         str(benchmark_result["savings_potential_kwh"]) + " kWh/yr | EUR " +
         str(benchmark_result["savings_potential_eur_year"]) + "/yr | " +
         str(benchmark_result["savings_potential_co2_kgCO2e"]) + " kgCO2e/yr"],
    ]
    t1 = Table(profile, colWidths=[6*cm, 11*cm])
    t1.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), BLUE),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME",      (0,1), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#F0F4FF")]),
        ("GRID",          (0,0), (-1,-1), 0.5, colors.lightgrey),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(KeepTogether([
        Paragraph("1. Building Profile", s_h2), t1
    ]))
    story.append(Spacer(1, 0.3*cm))
    story.append(KeepTogether([
        Paragraph("1b. EUI Benchmark Chart", s_h2),
        Image(chart_buf, width=14*cm, height=6*cm),
    ]))
    if dpe and dpe_badge_buf:
        dpe_badge_buf.seek(0)
        dpe_style = ParagraphStyle("dp", fontName="Helvetica-Bold", fontSize=11,
                                    textColor=colors.HexColor(dpe["final_color"]),
                                    spaceBefore=4, spaceAfter=4)
        story.append(KeepTogether([
            Paragraph("1c. DPE Energy Label", s_h2),
            Image(dpe_badge_buf, width=9*cm, height=7.5*cm),
            Paragraph(
                "Final DPE Label: " + dpe["final_class"] +
                "  |  " + str(dpe["ep_kWh_m2"]) + " kWh EP/m2/yr" +
                "  |  " + str(dpe["carbon_m2"]) + " kgCO2e/m2/yr", dpe_style),
        ]))
    ecm_data = [["ECM","kWh/yr","EUR/yr","kgCO2e/yr","Carbon Val\nEUR/yr","Adj\nPaybk","CapEx\nEUR","Payback","Priority"]]
    for e in ecm_list:
        ecm_data.append([e["ECM"], e["kWh/yr"], e["EUR/yr"],
                         e["kgCO2e/yr"],
                         e.get("Carbon Value EUR/yr",""),
                         e.get("Adj Payback",""),
                         e["CapEx EUR"], e["Payback"], e["Priority"]])
    total_kwh = sum(int(e["kWh/yr"].replace(",","")) for e in ecm_list)
    total_eur = sum(int(e["EUR/yr"].replace(",","")) for e in ecm_list)
    total_co2 = sum(int(e["kgCO2e/yr"].replace(",","")) for e in ecm_list)
    ecm_data.append(["TOTAL", str(total_kwh), str(total_eur), str(total_co2), "","","","",""])
    t2 = Table(ecm_data, colWidths=[3.6*cm,1.4*cm,1.4*cm,1.6*cm,1.5*cm,1.3*cm,1.5*cm,1.3*cm,1.1*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),  (-1,0),  BLUE),
        ("TEXTCOLOR",     (0,0),  (-1,0),  colors.white),
        ("FONTNAME",      (0,0),  (-1,0),  "Helvetica-Bold"),
        ("FONTNAME",      (0,-1), (-1,-1), "Helvetica-Bold"),
        ("BACKGROUND",    (0,-1), (-1,-1), colors.HexColor("#E8F5E9")),
        ("FONTSIZE",      (0,0),  (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),  (-1,-2), [colors.white, colors.HexColor("#F0F4FF")]),
        ("GRID",          (0,0),  (-1,-1), 0.5, colors.lightgrey),
        ("ALIGN",         (1,0),  (-1,-1), "RIGHT"),
        ("LEFTPADDING",   (0,0),  (-1,-1), 4),
        ("TOPPADDING",    (0,0),  (-1,-1), 3),
        ("BOTTOMPADDING", (0,0),  (-1,-1), 3),
    ]))
    story.append(KeepTogether([Paragraph("2. Energy Conservation Measures", s_h2), t2]))
    story.append(Paragraph("3. Executive Summary", s_h2))
    for line in narrative.split("\n"):
        line = clean(line)
        if not line:           story.append(Spacer(1, 3))
        elif line.isupper() and len(line) > 4:
                               story.append(Paragraph(line.title(), s_h3))
        elif line.startswith("- "):
                               story.append(Paragraph("- " + line[2:], s_bullet))
        else:                  story.append(Paragraph(line, s_body))
    comp_data = [["Regulation","Status","Obligation","Deadline"]]
    for f in compliance_flags:
        comp_data.append([
            Paragraph(f["regulation"], ParagraphStyle("rc", fontName="Helvetica",      fontSize=7.5, leading=10)),
            Paragraph(f["status"],     ParagraphStyle("rs", fontName="Helvetica-Bold", fontSize=7.5, leading=10)),
            Paragraph(f["obligation"], ParagraphStyle("ro", fontName="Helvetica",      fontSize=7.5, leading=10)),
            Paragraph(f["deadline"],   ParagraphStyle("rd", fontName="Helvetica",      fontSize=7.5, leading=10)),
        ])
    t3 = Table(comp_data, colWidths=[3.8*cm, 3.0*cm, 6.5*cm, 3.7*cm])
    style_cmds = [
        ("BACKGROUND",  (0,0), (-1,0), BLUE),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0), 8),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.lightgrey),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]
    for i, f in enumerate(compliance_flags, start=1):
        c = compliance_color(f["color"])
        style_cmds += [("BACKGROUND",(1,i),(1,i),c), ("TEXTCOLOR",(1,i),(1,i),colors.white)]
        if i % 2 == 0:
            style_cmds += [("BACKGROUND",(0,i),(0,i),colors.HexColor("#F0F4FF")),
                           ("BACKGROUND",(2,i),(3,i),colors.HexColor("#F0F4FF"))]
    t3.setStyle(TableStyle(style_cmds))
    story.append(KeepTogether([Paragraph("4. Regulatory Compliance Flags", s_h2), t3]))
    # Section 5 — Décret Tertiaire Trajectory
    if tertiaire_buf and tertiaire_targets:
        tertiaire_buf.seek(0)
        t5_rows = [
            ["Milestone", "Year", "Target EUI", "Required Reduction", "vs Best Practice"],
            ["OPERAT Reporting", "2026", "—", "First declaration", "—"],
            ["Target 1", "2030",
             str(round(tertiaire_targets[1], 1)) + " kWh/m2/yr",
             "-40% vs baseline",
             str(round(tertiaire_targets[1] - benchmark_result["benchmark_best"], 1)) + " kWh/m2/yr gap"],
            ["Target 2", "2040",
             str(round(tertiaire_targets[2], 1)) + " kWh/m2/yr",
             "-50% vs baseline",
             str(round(tertiaire_targets[2] - benchmark_result["benchmark_best"], 1)) + " kWh/m2/yr gap"],
            ["Target 3", "2050",
             str(round(tertiaire_targets[3], 1)) + " kWh/m2/yr",
             "-60% vs baseline",
             str(round(tertiaire_targets[3] - benchmark_result["benchmark_best"], 1)) + " kWh/m2/yr gap"],
        ]
        t5 = Table(t5_rows, colWidths=[3.5*cm, 1.8*cm, 3.2*cm, 3.5*cm, 3.5*cm])
        t5.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#0066CC")),
            ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#F0F4FF")]),
            ("GRID",          (0,0), (-1,-1), 0.5, colors.lightgrey),
            ("ALIGN",         (1,0), (-1,-1), "CENTER"),
            ("LEFTPADDING",   (0,0), (-1,-1), 5),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(KeepTogether([
            Paragraph("5. Decret Tertiaire 2019-771 Reduction Trajectory", s_h2),
            Image(tertiaire_buf, width=15*cm, height=7.5*cm),
        ]))
        story.append(Spacer(1, 0.2*cm))
        story.append(KeepTogether([
            Paragraph("Mandatory Milestones", s_h3),
            t5,
        ]))
    story.append(Spacer(1, 15))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Paragraph(
        ORG_NAME + " | " + ORG_TAGLINE,
        s_meta))
    doc.build(story)
    return path

def audit_one(bname, activity, floor_area, consumption, carrier, client, city="Paris"):
    b      = BENCHMARKS[activity]
    actual = consumption / floor_area
    cf     = CARBON_FACTORS[carrier]
    price  = ENERGY_PRICE[carrier]
    hdd_data = fetch_hdd(city if city else "Paris")
    weather   = weather_normalise(float(consumption),
                                  hdd_data["hdd_actual"],
                                  hdd_data["hdd_standard"])
    norm_kwh  = weather["normalised_kwh"]
    norm_eui  = round(norm_kwh / floor_area, 1)
    if actual <= b["eui_best"]:     tier = "Best Practice (Top 10%)"
    elif actual <= b["eui_median"]: tier = "Average"
    elif actual <= b["eui_p75"]:    tier = "Below Average"
    else:                           tier = "Poor (Bottom 25%)"
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
        "hdd_actual":        hdd_data["hdd_actual"],
        "hdd_standard":      HDD_STANDARD,
        "normalised_kwh":    norm_kwh,
        "carbon_liability_eur": round((float(consumption) * cf / 1000) * EU_ETS_PRICE),
    }
    ecms = []
    def add(name, share, pct, cost_m2, priority):
        kwh  = actual * floor_area * share * pct
        eur  = kwh * price
        cost = floor_area * cost_m2
        ecms.append({
            "ECM": name,
            "kWh/yr": str(round(kwh)), "EUR/yr": str(round(eur)),
            "kgCO2e/yr": str(round(kwh*cf)),
            "Carbon Value EUR/yr": str(round((kwh*cf/1000)*EU_ETS_PRICE)),
            "Adj Payback": (str(round(cost/(eur+(kwh*cf/1000)*EU_ETS_PRICE),1))+" yrs")
                           if (eur+(kwh*cf/1000)*EU_ETS_PRICE)>0 else "N/A",
            "CapEx EUR": str(round(cost)),
            "Payback": str(round(cost/eur, 1)) + " yrs", "Priority": priority,
            "_kwh": kwh, "_eur": eur, "_co2": kwh*cf,
        })
    add("BMS Optimisation",        1.00, 0.15,  8, "High")
    add("LED Lighting + Controls", 0.20, 0.60, 15, "High")
    if actual > b["eui_median"]:
        add("HVAC Heat Pump Upgrade", 1.00, 0.25, 80, "Medium")
    if actual > b["eui_median"] * 1.3:
        add("Building Envelope",      1.00, 0.10, 120, "Low")
    ecms.sort(key=lambda x: float(x["Payback"].split()[0]))
    ecm_text = "\n".join([
        "- " + e["ECM"] + ": " + e["kWh/yr"] + " kWh/yr, EUR " +
        e["EUR/yr"] + "/yr, " + e["Payback"] + " payback" for e in ecms])
    total_kwh = round(sum(e["_kwh"] for e in ecms))
    total_eur = round(sum(e["_eur"] for e in ecms))
    total_co2 = round(sum(e["_co2"] for e in ecms))
    msg = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=400,
        system=AXIOM_PROMPT,
        messages=[{"role": "user", "content":
            "Write a 150-word professional energy audit executive summary. "
            "Plain ASCII text only. Use ALL CAPS section headers. "
            "Building: " + bname + ", " + activity + ", " + str(floor_area) +
            " m2, EUI " + str(round(actual,1)) + " kWh/m2/yr, " + tier + "\n"
            "ECMs:\n" + ecm_text + "\nTotal: " + str(total_kwh) +
            " kWh/yr, EUR " + str(total_eur) + "/yr, " + str(total_co2) + " kgCO2e/yr"
        }]
    )
    narrative        = msg.content[0].text
    chart_buf        = make_chart(actual, b, activity)
    compliance_flags = check_compliance(floor_area, consumption)
    safe_name        = "".join(c if c.isalnum() else "_" for c in bname)
    pdf_path         = "/tmp/AXIOM_" + safe_name + ".pdf"
    carbon_m2_val  = actual * cf
    dpe_val        = get_dpe(actual, carrier, carbon_m2_val)
    dpe_badge_val  = make_dpe_badge(dpe_val)
    tertiaire_buf, tertiaire_targets = make_tertiaire_chart(actual, activity, floor_area)
    build_pdf(bname, benchmark_result, ecms, narrative,
              chart_buf, compliance_flags, pdf_path,
              dpe=dpe_val, dpe_badge_buf=dpe_badge_val,
              tertiaire_buf=tertiaire_buf, tertiaire_targets=tertiaire_targets)
    carbon_m2 = (consumption / floor_area) * cf
    dpe       = get_dpe(actual, carrier, carbon_m2)
    return {
        "Building": bname, "Activity": activity,
        "EUI": round(actual, 1), "Performance": tier,
        "DPE": dpe["final_class"],
        "Savings kWh/yr": total_kwh, "Savings EUR/yr": total_eur,
        "Savings kgCO2e/yr": total_co2,
        "Top ECM": ecms[0]["ECM"] if ecms else "",
        "Payback": ecms[0]["Payback"] if ecms else "",
        "_pdf": pdf_path,
        "_dpe": dpe,
    }

def run_audit(city, activity, floor_area, consumption, carrier):
    b      = BENCHMARKS[activity]
    actual = consumption / floor_area
    cf     = CARBON_FACTORS[carrier]
    price  = ENERGY_PRICE[carrier]
    hdd_data  = fetch_hdd(city if city else "Paris")
    weather   = weather_normalise(float(consumption),
                                  hdd_data["hdd_actual"],
                                  hdd_data["hdd_standard"])
    norm_kwh  = weather["normalised_kwh"]
    norm_eui  = round(norm_kwh / floor_area, 1)
    if actual <= b["eui_best"]:     tier = "Best Practice (Top 10%)"
    elif actual <= b["eui_median"]: tier = "Average"
    elif actual <= b["eui_p75"]:    tier = "Below Average"
    else:                           tier = "Poor (Bottom 25%)"
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
        "hdd_actual":        hdd_data["hdd_actual"],
        "hdd_standard":      HDD_STANDARD,
        "normalised_kwh":    norm_kwh,
        "carbon_liability_eur": round((float(consumption) * cf / 1000) * EU_ETS_PRICE),
    }
    ecms = []
    def add(name, share, pct, cost_m2, priority):
        kwh  = actual * floor_area * share * pct
        eur  = kwh * price
        cost = floor_area * cost_m2
        ecms.append({
            "ECM": name,
            "kWh/yr": str(round(kwh)), "EUR/yr": str(round(eur)),
            "kgCO2e/yr": str(round(kwh*cf)),
            "Carbon Value EUR/yr": str(round((kwh*cf/1000)*EU_ETS_PRICE)),
            "Adj Payback": (str(round(cost/(eur+(kwh*cf/1000)*EU_ETS_PRICE),1))+" yrs")
                           if (eur+(kwh*cf/1000)*EU_ETS_PRICE)>0 else "N/A",
            "CapEx EUR": str(round(cost)),
            "Payback": str(round(cost/eur, 1)) + " yrs", "Priority": priority,
            "_kwh": kwh, "_eur": eur, "_co2": kwh*cf,
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
    benchmark_md = "## Benchmark Results\n| Metric | Value |\n|--------|-------|\n"
    benchmark_md += "| **Actual EUI** | " + str(round(actual,1)) + " kWh/m2/yr |\n"
    benchmark_md += "| **Normalised EUI** | " + str(norm_eui) + " kWh/m2/yr (" + str(hdd_data['hdd_actual']) + " HDD → std 2500) |\n"
    benchmark_md += "| **Sector Median** | " + str(b["eui_median"]) + " kWh/m2/yr |\n"
    benchmark_md += "| **Best Practice** | " + str(b["eui_best"]) + " kWh/m2/yr |\n"
    benchmark_md += "| **Performance** | " + tier + " |\n"
    benchmark_md += "| **Savings** | " + str(sav_kwh) + " kWh/yr / EUR " + str(sav_eur) + "/yr / " + str(sav_co2) + " kgCO2e/yr |\n"
    df     = pd.DataFrame([{k:v for k,v in e.items() if not k.startswith("_")} for e in ecms])
    ecm_md = "## ECM Priority List\n" + df.to_markdown(index=False)
    ecm_md += "\n\n**Total:** " + str(total_kwh) + " kWh/yr / EUR " + str(total_eur) + "/yr / " + str(total_co2) + " kgCO2e/yr"
    client   = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    ecm_text = "\n".join([
        "- " + e["ECM"] + ": " + e["kWh/yr"] + " kWh/yr, EUR " +
        e["EUR/yr"] + "/yr, " + e["Payback"] + " payback" for e in ecms])
    msg = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=500,
        system=AXIOM_PROMPT,
        messages=[{"role": "user", "content":
            "Write a 200-word professional energy audit executive summary. "
            "Plain ASCII text only. No markdown. Use ALL CAPS section headers. "
            "Building: " + activity + ", " + str(floor_area) + " m2, EUI " +
            str(round(actual,1)) + " kWh/m2/yr, performance: " + tier + "\n"
            "ECMs:\n" + ecm_text + "\n"
            "Total: " + str(total_kwh) + " kWh/yr, EUR " + str(total_eur) + "/yr, " + str(total_co2) + " kgCO2e/yr"
        }]
    )
    narrative        = msg.content[0].text
    chart_buf        = make_chart(actual, b, activity)
    compliance_flags = check_compliance(floor_area, consumption)
    carbon_m2        = actual * cf
    dpe              = get_dpe(actual, carrier, carbon_m2)
    dpe_badge_buf    = make_dpe_badge(dpe)
    tertiaire_buf, tertiaire_targets = make_tertiaire_chart(actual, activity, floor_area)
    bname    = activity + " | " + str(floor_area) + " m2 | " + carrier
    pdf_path = build_pdf(bname, benchmark_result, ecms, narrative,
                         chart_buf, compliance_flags, "/tmp/AXIOM_Report.pdf",
                         dpe=dpe, dpe_badge_buf=dpe_badge_buf,
                         tertiaire_buf=tertiaire_buf, tertiaire_targets=tertiaire_targets)
    comp_md = "## Regulatory Compliance\n| Regulation | Status | Deadline |\n|-----------|--------|---------|\n"
    for f in compliance_flags:
        comp_md += "| " + f["regulation"] + " | **" + f["status"] + "** | " + f["deadline"] + " |\n"
    dpe_md = (
        "\n\n## DPE Label\n"
        "| Metric | Value |\n|--------|-------|\n"
        "| **DPE Energy Class** | **" + dpe["energy_class"] + "** |\n"
        "| **DPE GHG Class** | **" + dpe["ghg_class"] + "** |\n"
        "| **Final DPE Label** | **" + dpe["final_class"] + "** |\n"
        "| Primary Energy | " + str(dpe["ep_kWh_m2"]) + " kWh EP/m2/yr |\n"
        "| Carbon Intensity | " + str(dpe["carbon_m2"]) + " kgCO2e/m2/yr |\n"
    )
    t1_eui = tertiaire_targets[1]
    t2_eui = tertiaire_targets[2]
    t3_eui = tertiaire_targets[3]
    tertiaire_md = (
        "\n\n## Decret Tertiaire 2019-771\n"
        "| Milestone | Target EUI | Reduction |\n|-----------|-----------|-----------|\n"
        "| **2030** | " + str(round(t1_eui,1)) + " kWh/m2/yr | -40% vs baseline |\n"
        "| **2040** | " + str(round(t2_eui,1)) + " kWh/m2/yr | -50% vs baseline |\n"
        "| **2050** | " + str(round(t3_eui,1)) + " kWh/m2/yr | -60% vs baseline |\n"
        "\n*OPERAT reporting deadline: 30 September 2026*"
    )
    carbon_eur = benchmark_result.get("carbon_liability_eur", 0)
    weather_md = (
        "\n\n## Weather Normalisation\n"
        "| Metric | Value |\n|--------|-------|\n"
        "| City | " + str(city) + " |\n"
        "| Actual HDD (base 18°C) | " + str(weather["hdd_actual"]) + " |\n"
        "| Standard HDD | " + str(weather["hdd_standard"]) + " |\n"
        "| Raw consumption | " + str(round(float(consumption))) + " kWh |\n"
        "| Normalised consumption | " + str(weather["normalised_kwh"]) + " kWh |\n"
        "| Correction | " + str(weather["correction_pct"]) + "% |\n"
        "\n> " + weather["weather_note"] + "\n"
    )
    weather_md = (
        "\n\n## Weather Normalisation\n"
        "| Metric | Value |\n|--------|-------|\n"
        "| City | " + str(city) + " |\n"
        "| Actual HDD (base 18°C) | " + str(weather["hdd_actual"]) + " |\n"
        "| Standard HDD | " + str(weather["hdd_standard"]) + " |\n"
        "| Raw consumption | " + str(round(float(consumption))) + " kWh |\n"
        "| Normalised consumption | " + str(norm_kwh) + " kWh |\n"
        "| Correction | " + str(weather["correction_pct"]) + "% |\n"
        "\n> " + weather["weather_note"] + "\n"
    )
    carbon_md  = (
        "\n\n## Carbon Pricing (EU ETS)\n"
        "| Metric | Value |\n|--------|-------|\n"
        "| EU ETS Price | EUR " + str(int(EU_ETS_PRICE)) + "/tCO2e |\n"
        "| Annual Carbon Liability | EUR " + str(carbon_eur) + "/yr |\n"
        "| Carbon saved (all ECMs) | EUR " + str(round(total_co2/1000*EU_ETS_PRICE)) + "/yr |\n"
    )
    return benchmark_md + dpe_md + tertiaire_md + carbon_md + weather_md, ecm_md, narrative, comp_md, pdf_path



def _pc(text, size=7, bold=False):
    """Wrap text in a ReportLab Paragraph so it word-wraps inside table cells."""
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    style = ParagraphStyle(
        "cell",
        fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=size,
        leading=size + 2,
        alignment=TA_LEFT,
        wordWrap="CJK",
        leftPadding=0, rightPadding=0,
        spaceBefore=0, spaceAfter=0,
    )
    return Paragraph(str(text), style)

def _pc_hdr(text, size=7):
    """Bold white header cell."""
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    style = ParagraphStyle(
        "hdr",
        fontName="Helvetica-Bold",
        fontSize=size,
        leading=size + 2,
        alignment=TA_CENTER,
        textColor=colors.white,
        wordWrap="CJK",
        leftPadding=0, rightPadding=0,
    )
    return Paragraph(str(text), style)

def build_portfolio_pdf(results, dash_buf, path):
    """One-page Portfolio Summary PDF: dashboard chart + KPI table."""
    BLUE = colors.HexColor("#0066CC")
    DARK = colors.HexColor("#1A1A2E")
    doc  = SimpleDocTemplate(path, pagesize=landscape(A4),
                              rightMargin=1.5*cm, leftMargin=1.5*cm,
                              topMargin=2*cm, bottomMargin=1.5*cm)
    s_title = ParagraphStyle("t",  fontName="Helvetica-Bold", fontSize=16,
                              textColor=DARK, alignment=TA_CENTER, spaceAfter=4)
    s_meta  = ParagraphStyle("m",  fontName="Helvetica", fontSize=7.5,
                              textColor=colors.grey, alignment=TA_CENTER, spaceAfter=6)
    s_h2    = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10,
                              textColor=BLUE, spaceBefore=10, spaceAfter=5)
    story = []
    story.append(Paragraph("AXIOM Portfolio Energy Audit", s_title))
    story.append(Paragraph(
        "Generated: " + date.today().strftime("%d %B %Y") +
        " | " + str(len(results)) + " buildings | ISO 50002:2014 | EU EED 2023/1791",
        s_meta))
    story.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=8))

    # Dashboard image — full width
    story.append(Paragraph("Portfolio Dashboard", s_h2))
    dash_buf.seek(0)
    story.append(Image(dash_buf, width=18*cm, height=11.25*cm))
    story.append(Spacer(1, 0.3*cm))

    # KPI summary table
    story.append(KeepTogether([
        Paragraph("Building Summary", s_h2),
    ]))
    header = [_pc_hdr("Building"), _pc_hdr("Activity"),
              _pc_hdr("EUI\nkWh/m2/yr"), _pc_hdr("DPE"),
              _pc_hdr("Performance"), _pc_hdr("Savings\nkWh/yr"),
              _pc_hdr("Savings\nEUR/yr"), _pc_hdr("Top ECM"), _pc_hdr("Payback")]
    rows   = [header]
    total_kwh = 0
    total_eur = 0
    for r in results:
        rows.append([
            _pc(r["Building"]),
            _pc(r["Activity"]),
            _pc(str(r["EUI"])),
            _pc(r.get("DPE", "")),
            _pc(r["Performance"]),
            _pc(str(r["Savings kWh/yr"])),
            _pc("EUR " + str(r["Savings EUR/yr"])),
            _pc(r["Top ECM"]),
            _pc(r["Payback"]),
        ])
        total_kwh += r["Savings kWh/yr"]
        total_eur += r["Savings EUR/yr"]
    rows.append([_pc("TOTAL", bold=True), _pc(""), _pc(""), _pc(""), _pc(""),
                 _pc(str(total_kwh), bold=True),
                 _pc("EUR " + str(total_eur), bold=True),
                 _pc(""), _pc("")])

    tier_bg = {
        "Best Practice (Top 10%)": colors.HexColor("#D5F5E3"),
        "Average":                  colors.HexColor("#FEF9E7"),
        "Below Average":            colors.HexColor("#FDEBD0"),
        "Poor (Bottom 25%)":        colors.HexColor("#FADBD8"),
    }

    t = Table(rows, colWidths=[3.8*cm, 3.2*cm, 1.6*cm, 1.1*cm, 3.2*cm,
                                2.0*cm, 2.0*cm, 2.8*cm, 1.6*cm])
    style_cmds = [
        ("BACKGROUND",    (0,0), (-1,0), BLUE),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME",      (0,-1),(-1,-1),"Helvetica-Bold"),
        ("BACKGROUND",    (0,-1),(-1,-1),colors.HexColor("#E8F5E9")),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.lightgrey),
        ("ALIGN",         (2,0), (-1,-1), "RIGHT"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("WORDPADDING",   (0,0), (-1,-1), 1),
        ("LEFTPADDING",   (0,0), (-1,-1), 4),
        ("RIGHTPADDING",  (0,0), (-1,-1), 4),
        ("TOPPADDING",    (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("ROWBACKGROUNDS",(0,1), (-1,-2), [colors.white, colors.HexColor("#F0F4FF")]),
    ]
    # Colour performance cells
    for i, r in enumerate(results, start=1):
        bg = tier_bg.get(r["Performance"], colors.white)
        style_cmds.append(("BACKGROUND", (3,i), (3,i), bg))
    t.setStyle(TableStyle(style_cmds))
    story.append(Spacer(1, 6))
    story.append(KeepTogether([t]))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Paragraph(
        ORG_NAME + " | " + ORG_TAGLINE,
        s_meta))
    doc.build(story)
    return path

def run_batch(uploaded_file):
    if uploaded_file is None:
        return "⬆️ Please upload a file (CSV, XLSX, ODS or PDF).", None, None
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    df, err = read_input_file(uploaded_file, client)
    if err:
        return "❌ Error: " + err + "\n\nPlease fix your file and upload again.", None, None
    if df is None or df.empty:
        return "❌ No data found. Please check format and upload again.", None, None
    required = {"building_name","activity","floor_area_m2","consumption_kwh","energy_carrier"}
    missing  = required - set(df.columns)
    if missing:
        return (
            "❌ Missing columns: **" + ", ".join(missing) + "**\n\n"
            "Your file has: " + ", ".join(df.columns.tolist()) + "\n\n"
            "See CSV template on the left.", None, None
        )
    results = []
    errors  = []
    for _, row in df.iterrows():
        try:
            bname    = str(row["building_name"])
            activity = str(row["activity"]).strip()
            carrier  = str(row["energy_carrier"]).strip()
            if activity not in BENCHMARKS:
                errors.append("**" + bname + "**: unknown activity '" + activity + "'")
                continue
            if carrier not in CARBON_FACTORS:
                errors.append("**" + bname + "**: unknown carrier '" + carrier + "'")
                continue
            r = audit_one(bname, activity, int(row["floor_area_m2"]),
              int(row["consumption_kwh"]), carrier, client, city="Paris")
            results.append(r)
        except Exception as e:
            errors.append("**" + str(row.get("building_name","?")) + "**: " + str(e))
    if not results:
        return "❌ No buildings audited.\n\n" + "\n".join(errors), None, None

    summary_df = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in results])
    total_kwh  = summary_df["Savings kWh/yr"].sum()
    total_eur  = summary_df["Savings EUR/yr"].sum()
    total_co2  = summary_df["Savings kgCO2e/yr"].sum()
    portfolio_md  = "## ✅ Portfolio Audit — " + str(len(results)) + " buildings\n"
    portfolio_md += summary_df.to_markdown(index=False)
    portfolio_md += (
        "\n\n**Portfolio Total:** " + str(total_kwh) +
        " kWh/yr | EUR " + str(total_eur) + "/yr | " + str(total_co2) + " kgCO2e/yr"
    )
    if errors:
        portfolio_md += "\n\n⚠️ **Skipped:** " + " | ".join(errors)

    # Generate dashboard chart
    dash_buf = make_portfolio_dashboard(results)
    dash_path = "/tmp/AXIOM_Dashboard.png"
    with open(dash_path, "wb") as f:
        f.write(dash_buf.read())

    # Build Portfolio Summary PDF
    dash_buf.seek(0)
    import copy
    dash_buf2 = io.BytesIO(dash_buf.getvalue())
    portfolio_pdf_path = "/tmp/AXIOM_Portfolio_Summary.pdf"
    build_portfolio_pdf(results, dash_buf2, portfolio_pdf_path)

    zip_path = "/tmp/AXIOM_Portfolio_Audit.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(portfolio_pdf_path, "AXIOM_Portfolio_Summary.pdf")
        for r in results:
            zf.write(r["_pdf"], os.path.basename(r["_pdf"]))
        zf.write(dash_path, "AXIOM_Portfolio_Dashboard.png")

    return portfolio_md, dash_path, zip_path


# ─────────────────────────────────────────────────────────────────────────────
# ESCO FINANCING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def esco_analysis(
    capex_eur: float,
    annual_savings_eur: float,
    annual_co2_kg: float,
    contract_years: int = 10,
    esco_share: float = 0.70,
    discount_rate: float = 0.06,
    loan_rate: float = 0.045,
    carbon_price: float = EU_ETS_PRICE,
) -> dict:
    """
    Model three financing structures for an ECM package:
      1. ESCO shared-savings (client gets 1-esco_share of savings)
      2. Bank loan (flat annual repayment)
      3. Self-funded (full savings from year 1)
    Returns cash flows, NPV, IRR, and payback for each structure.
    """
    carbon_value_yr = round((annual_co2_kg / 1000) * carbon_price)
    total_savings   = annual_savings_eur + carbon_value_yr

    # ── ESCO shared savings ──────────────────────────────────────────────────
    client_share   = 1 - esco_share
    client_cf      = [round(total_savings * client_share) for _ in range(contract_years)]
    post_contract  = [round(total_savings) for _ in range(5)]  # full savings after
    esco_cf        = [round(total_savings * esco_share) for _ in range(contract_years)]
    esco_npv       = round(sum(v / (1+discount_rate)**t
                               for t, v in enumerate(esco_cf, 1)) - capex_eur)
    client_npv     = round(sum(v / (1+discount_rate)**t
                               for t, v in enumerate(client_cf + post_contract, 1)))

    # ── Bank loan ────────────────────────────────────────────────────────────
    annual_payment = round(capex_eur * loan_rate /
                           (1 - (1 + loan_rate)**-contract_years))
    loan_net_cf    = [round(total_savings - annual_payment)
                      for _ in range(contract_years)]
    loan_npv       = round(sum(v / (1+discount_rate)**t
                               for t, v in enumerate(loan_net_cf, 1)) - 0)

    # ── Self-funded ──────────────────────────────────────────────────────────
    self_cf        = [-capex_eur] + [round(total_savings)
                                      for _ in range(contract_years + 4)]
    self_npv       = round(sum(v / (1+discount_rate)**t
                               for t, v in enumerate(self_cf)))
    simple_payback = round(capex_eur / total_savings, 1) if total_savings > 0 else 99

    # ── IRR via Newton (self-funded) ─────────────────────────────────────────
    def irr(cashflows, guess=0.1):
        r = guess
        for _ in range(100):
            npv  = sum(cf / (1+r)**t for t, cf in enumerate(cashflows))
            dnpv = sum(-t * cf / (1+r)**(t+1) for t, cf in enumerate(cashflows))
            if abs(dnpv) < 1e-10:
                break
            r -= npv / dnpv
        return round(r * 100, 1)

    self_irr = irr(self_cf)

    # ── Year-by-year table (first 10 yrs) ────────────────────────────────────
    rows = []
    loan_balance = capex_eur
    for yr in range(1, contract_years + 1):
        interest      = round(loan_balance * loan_rate)
        principal     = round(annual_payment - interest)
        loan_balance  = max(0, round(loan_balance - principal))
        rows.append({
            "Year":          yr,
            "Gross Savings": total_savings,
            "ESCO (client)": client_cf[yr-1],
            "Loan net CF":   loan_net_cf[yr-1],
            "Self-fund CF":  round(total_savings),
            "Loan balance":  loan_balance,
        })

    return {
        "capex":            capex_eur,
        "annual_savings":   annual_savings_eur,
        "carbon_value_yr":  carbon_value_yr,
        "total_savings_yr": total_savings,
        "contract_years":   contract_years,
        "esco_share_pct":   round(esco_share * 100),
        "client_share_pct": round(client_share * 100),
        "esco_npv":         esco_npv,
        "client_npv":       client_npv,
        "loan_npv":         loan_npv,
        "self_npv":         self_npv,
        "self_irr":         self_irr,
        "simple_payback":   simple_payback,
        "annual_payment":   annual_payment,
        "cashflow_table":   rows,
        "discount_rate":    discount_rate,
        "loan_rate":        loan_rate,
    }

def format_esco_md(r: dict) -> str:
    """Format ESCO analysis dict as Markdown for the UI."""
    md  = "## ESCO Financing Analysis\n\n"
    md += "### Project Inputs\n"
    md += f"| Parameter | Value |\n|-----------|-------|\n"
    md += f"| CapEx | EUR {r['capex']:,.0f} |\n"
    md += f"| Annual Energy Savings | EUR {r['annual_savings']:,.0f}/yr |\n"
    md += f"| Carbon Value (EU ETS @EUR {int(EU_ETS_PRICE)}/tCO2e) | EUR {r['carbon_value_yr']:,.0f}/yr |\n"
    md += f"| **Total Annual Benefit** | **EUR {r['total_savings_yr']:,.0f}/yr** |\n"
    md += f"| Simple Payback | {r['simple_payback']} yrs |\n"
    md += f"| Project IRR (self-funded) | {r['self_irr']}% |\n\n"

    md += "### Financing Structure Comparison\n"
    md += "| Structure | Client Annual CF | NPV (10yr) | Notes |\n"
    md += "|-----------|-----------------|------------|-------|\n"
    md += (f"| **ESCO Shared Savings** | EUR {int(r['total_savings_yr']*r['client_share_pct']/100):,}/yr "
           f"| EUR {r['client_npv']:,} | ESCO funds CapEx; {r['esco_share_pct']}% to ESCO |\n")
    md += (f"| **Bank Loan** | EUR {r['total_savings_yr'] - r['annual_payment']:,}/yr "
           f"| EUR {r['loan_npv']:,} | Annual payment EUR {r['annual_payment']:,} |\n")
    md += (f"| **Self-Funded** | EUR {r['total_savings_yr']:,}/yr "
           f"| EUR {r['self_npv']:,} | IRR {r['self_irr']}% — best long-term return |\n\n")

    md += "### Annual Cash Flow Table\n"
    md += "| Year | Gross Savings | ESCO Client Share | Loan Net CF | Self-Fund CF | Loan Balance |\n"
    md += "|------|--------------|-------------------|-------------|--------------|--------------|\n"
    for row in r["cashflow_table"]:
        md += (f"| {row['Year']} | EUR {row['Gross Savings']:,} "
               f"| EUR {row['ESCO (client)']:,} "
               f"| EUR {row['Loan net CF']:,} "
               f"| EUR {row['Self-fund CF']:,} "
               f"| EUR {row['Loan balance']:,} |\n")

    md += "\n> **Recommendation:** "
    if r["self_irr"] > 15:
        md += f"IRR of {r['self_irr']}% strongly favours self-funding if capital is available."
    elif r["simple_payback"] <= 7:
        md += f"Simple payback of {r['simple_payback']} yrs supports bank loan — lowest total cost."
    else:
        md += f"Payback of {r['simple_payback']} yrs favours ESCO structure — zero CapEx risk for client."
    return md

def run_esco(capex, savings_eur, co2_kg, years, esco_pct, discount, loan_rate):
    try:
        r  = esco_analysis(
            capex_eur=float(capex),
            annual_savings_eur=float(savings_eur),
            annual_co2_kg=float(co2_kg),
            contract_years=int(years),
            esco_share=float(esco_pct)/100,
            discount_rate=float(discount)/100,
            loan_rate=float(loan_rate)/100,
        )
        return format_esco_md(r)
    except Exception as ex:
        return f"❌ Error: {ex}"

with gr.Blocks(title=ORG_NAME, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# AXIOM Energy Audit Platform
### Automated EU-compliant energy audits powered by Claude AI
*Benchmarked against 1M+ French tertiary buildings (ADEME open data)*
---
""")
    with gr.Tabs():
        with gr.Tab("Single Building Audit"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Building Profile")
                    city        = gr.Dropdown(list(HDD_REFERENCE.keys()),
                                     value="Paris", label="📍 City (weather normalisation)")
                    activity    = gr.Dropdown(list(BENCHMARKS.keys()), value="Bureaux",
                                              label="Activity Type")
                    floor_area  = gr.Slider(500, 50000, value=5000,
                                            step=500, label="Floor Area (m2)")
                    consumption = gr.Slider(50000, 5000000, value=800000,
                                            step=10000, label="Annual Consumption (kWh)")
                    carrier     = gr.Dropdown(list(CARBON_FACTORS.keys()),
                                              value="Electricite", label="Energy Carrier")
                    btn_single  = gr.Button("Run AXIOM Audit", variant="primary", size="lg")
                with gr.Column(scale=2):
                    benchmark_out = gr.Markdown()
                    ecm_out       = gr.Markdown()
            gr.Markdown("### Executive Summary")
            narrative_out  = gr.Textbox(label="", lines=6)
            gr.Markdown("### Regulatory Compliance")
            compliance_out = gr.Markdown()
            gr.Markdown("### Download Report")
            pdf_out = gr.File(label="AXIOM_Report.pdf", file_types=[".pdf"])
            btn_single.click(
                fn=run_audit,
                inputs=[city, activity, floor_area, consumption, carrier],
                outputs=[benchmark_out, ecm_out, narrative_out, compliance_out, pdf_out]
            )

        with gr.Tab("💰 ESCO / Financing"):
            gr.Markdown("""
## ESCO & Project Financing Calculator
Model three financing structures for any ECM package: **ESCO shared-savings**, **bank loan**, or **self-funded**.
Carbon value (EU ETS) is included in the benefit stream automatically.
""")
            with gr.Row():
                with gr.Column(scale=1):
                    esco_capex    = gr.Number(label="Total CapEx (EUR)",                   value=100000)
                    esco_savings  = gr.Number(label="Annual Energy Savings (EUR/yr)",       value=15000)
                    esco_co2      = gr.Number(label="Annual CO₂ Reduction (kgCO₂e/yr)",    value=30000)
                with gr.Column(scale=1):
                    esco_years    = gr.Slider(5, 20, value=10, step=1,
                                              label="Contract Length (years)")
                    esco_pct      = gr.Slider(50, 90, value=70, step=5,
                                              label="ESCO Share of Savings (%)")
                    esco_discount = gr.Number(label="Discount Rate (%)",      value=6.0)
                    esco_loan     = gr.Number(label="Loan Interest Rate (%)", value=4.5)
            btn_esco = gr.Button("Run ESCO Analysis", variant="primary", size="lg")
            esco_out = gr.Markdown()
            btn_esco.click(run_esco,
                inputs=[esco_capex, esco_savings, esco_co2,
                        esco_years, esco_pct, esco_discount, esco_loan],
                outputs=[esco_out])
        with gr.Tab("🎨 Branding & Settings"):
            gr.Markdown("""
## Platform Branding
Customise your organisation name, tagline, and accent colour via **HF Space Secrets**:

| Secret Key | Purpose | Example |
|-----------|---------|---------|
| `AXIOM_USER` | Login username | `auditor` |
| `AXIOM_PASS` | Login password | `MySecurePass123` |
| `AXIOM_ORG_NAME` | Organisation name on PDF cover | `Engie Services` |
| `AXIOM_TAGLINE` | PDF footer tagline | `ISO 50002 Certified Auditors` |
| `AXIOM_COLOR` | Accent colour (hex) | `#E63946` |

> Set secrets at: **HF Space → Settings → Variables and Secrets**
> Changes apply on next Space restart.
""")
            with gr.Row():
                gr.Textbox(label="Current Org Name",    value=ORG_NAME,    interactive=False)
                gr.Textbox(label="Current Tagline",     value=ORG_TAGLINE, interactive=False)
                gr.Textbox(label="Current Brand Colour",value=BRAND_COLOR, interactive=False)
            gr.Markdown("""
### Auth Status
""")
            auth_status = gr.Textbox(
                label="Auth Mode",
                value="🔒 Password protected" if os.environ.get("AXIOM_USER")
                      else "🔓 Open access (set AXIOM_USER + AXIOM_PASS secrets to enable)",
                interactive=False
            )
        with gr.Tab("Batch Portfolio Audit"):
            gr.Markdown("""
### Upload a portfolio file to audit multiple buildings at once
Supported formats: **CSV · XLSX · ODS · PDF**  —  One PDF per building + dashboard chart, all in a ZIP.

**Required columns:** `building_name` | `activity` | `floor_area_m2` | `consumption_kwh` | `energy_carrier`
""")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**CSV Template** — copy, fill in your buildings, save as .csv")
                    csv_template = gr.Textbox(
                        value=BATCH_TEMPLATE, label="", lines=8, interactive=False)
                with gr.Column(scale=1):
                    file_upload = gr.File(
                        label="Upload Portfolio (CSV / XLSX / ODS / PDF)",
                        file_types=[".csv", ".xlsx", ".xls", ".ods", ".pdf"],
                        file_count="single",
                    )
                    btn_batch = gr.Button("Run Batch Audit", variant="primary", size="lg")
            portfolio_out = gr.Markdown(
                value="Upload a file above and click **Run Batch Audit** to begin.")
            gr.Markdown("### Portfolio Dashboard")
            dashboard_img = gr.Image(label="", type="filepath")
            gr.Markdown("### Download All Reports")
            zip_out = gr.File(label="Download Reports + Dashboard (ZIP)")
            btn_batch.click(
                fn=run_batch,
                inputs=[file_upload],
                outputs=[portfolio_out, dashboard_img, zip_out]
            )

    gr.Markdown("---\n*AXIOM | ISO 50002:2014 | EU EED 2023/1791 | Powered by Claude AI*")

demo.launch(auth=_check_password if os.environ.get("AXIOM_USER") else None,
            auth_message="Enter your AXIOM credentials to continue.")
