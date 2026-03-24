# AXIOM — Automated Energy Audit Platform

> **AI-powered energy auditing for buildings and portfolios — ISO 50002 compliant, EU EED ready, PDF reports in seconds.**

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/Mr-MB/axiom-energy-audit)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Claude AI](https://img.shields.io/badge/Powered%20by-Claude%20AI-purple)](https://anthropic.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🔗 Try It Now

**[https://huggingface.co/spaces/Mr-MB/axiom-energy-audit](https://huggingface.co/spaces/Mr-MB/axiom-energy-audit)**

No installation required. Open in your browser, upload a CSV or fill in building details, and get a full ISO 50002 audit report with PDF download in under 30 seconds.

---

## What Is AXIOM?

AXIOM is a fully automated energy audit platform that replicates and accelerates the work of a qualified energy auditor. It ingests building data, benchmarks performance against sector medians, identifies Energy Conservation Measures (ECMs), models financial and carbon returns, checks regulatory compliance, and generates a professional PDF report — all powered by Claude AI.

### Standards & Compliance Coverage

| Standard | Coverage |
|----------|----------|
| **ISO 50002:2014** | Full audit methodology |
| **EU EED 2023/1791** | Art. 11 mandatory audit trigger |
| **Décret Tertiaire 2019-771** | -40/-50/-60% trajectory + OPERAT |
| **DPE (France)** | A–G energy + GES label |
| **EU ETS** | Carbon pricing @€65/tCO₂e |
| **CSRD / ESRS E1** | Scope 1/2/3 compliance flag |
| **ISO 50001:2018** | EnMS recommendation |
| **IPMVP (EVO 10000)** | M&V protocol recommendation |

---

## ✨ Features

### 🏢 Single Building Audit
- EUI benchmarking against sector medians (15 activity types)
- Performance tier: Best Practice / Average / Below Average / Poor
- 4 ECMs auto-generated: BMS, LED, HVAC Heat Pump, Building Envelope
- **Carbon-adjusted payback** — EU ETS value included in ECM financials
- **Weather normalisation** — Open-Meteo HDD (base 18°C), 12 French cities
- **DPE A–G badge** with GES class subplot
- **Décret Tertiaire** trajectory chart — 2030/2040/2050 milestones
- Claude AI executive summary (ISO 50002, plain-language)
- Full regulatory compliance table (6 regulations)
- Professional PDF report download

### 📦 Batch Portfolio Audit
- Upload CSV / XLSX / ODS / PDF with multiple buildings
- Processes entire portfolio in one click
- Portfolio dashboard: EUI benchmark chart, savings bar chart, tier donut
- Building summary table with DPE, performance, savings, payback
- Portfolio PDF with landscape summary table
- ZIP download with all individual building PDFs

### 💰 ESCO / Financing Calculator
- Three financing structures: **ESCO shared savings**, **bank loan**, **self-funded**
- IRR (Newton-Raphson), NPV at configurable discount rate
- EU ETS carbon value auto-included in benefit stream
- 10-year cash flow table with loan balance
- Smart recommendation engine (IRR / payback decision logic)

### 🎨 Branding & Auth
- Password protection via HF Secrets (`AXIOM_USER` / `AXIOM_PASS`)
- Custom org name, tagline, accent colour on PDF cover and footer
- Timing-safe authentication (SHA-256 + `hmac.compare_digest`)

---

## 🧪 Test Cases

Use these to test the Single Building Audit tab:

| Building | Activity | Area (m²) | Consumption (kWh/yr) | Carrier | Expected DPE |
|----------|----------|-----------|---------------------|---------|-------------|
| Office Paris | Bureaux | 5 000 | 875 000 | Electricite | C |
| Primary School | Enseignement Primaire | 2 800 | 245 000 | Gaz | C |
| Hospital | Sante - Centres Hospitaliers | 12 000 | 3 600 000 | Electricite | G |
| Warehouse | Logistique | 20 000 | 600 000 | Fioul | B |
| Hotel | Hotellerie | 6 000 | 880 000 | Electricite | F |

### Batch CSV Template

```csv
building_name,activity,floor_area_m2,consumption_kwh,energy_carrier
Tour Montparnasse,Bureaux,15000,2100000,Electricite
Lycee Victor Hugo,Enseignement Secondaire,8000,720000,Gaz
Hotel Mercure Lyon,Hotellerie,6000,950000,Electricite
Entrepot Rungis,Logistique,25000,800000,Fioul
Clinique Saint-Louis,Sante - Centres Hospitaliers,12000,3200000,Electricite
```

### Valid Energy Carriers
`Electricite` | `Gaz` | `Fioul` | `Reseau de chaleur` | `Reseau de froid` | `Bois`

### Valid Activity Types
`Bureaux` | `Enseignement Primaire` | `Enseignement Secondaire` | `Enseignement Superieur` | `Sante - Centres Hospitaliers` | `Hotellerie` | `Logistique` | `Commerce` | `Sports` | `Restauration` | `Culturel` | `Judiciaire` | `Penitentiaire` | `Social - Hebergement` | `Autre`

---

## 🏗️ Architecture

```
axiom-energy-audit-platform/
├── app.py                  # Main Gradio application (single file, ~1100 lines)
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata
├── docs/                   # Architecture docs
├── notebooks/              # Jupyter exploration notebooks
├── src/                    # Modular source (future refactor)
└── tests/                  # Test suite
```

### Tech Stack

| Layer | Technology |
|-------|------------|
| **UI** | Gradio 4.x (Soft theme) |
| **AI** | Anthropic Claude (claude-sonnet-4-5) |
| **PDF** | ReportLab (platypus) |
| **Charts** | Matplotlib |
| **Data** | Pandas, NumPy |
| **Weather** | Open-Meteo Archive API |
| **Hosting** | Hugging Face Spaces |
| **Auth** | HF Secrets + hmac/SHA-256 |

---

## 🚀 Self-Hosting / Local Setup

```bash
# 1. Clone
git clone https://github.com/mike-brady1/AXIOM-energy-audit-platform.git
cd AXIOM-energy-audit-platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Run
python app.py
# → Open http://localhost:7860
```

### HF Space Secrets (Optional)

| Secret | Purpose | Default |
|--------|---------|--------|
| `ANTHROPIC_API_KEY` | Claude AI — **required** | — |
| `AXIOM_USER` | Login username | open access |
| `AXIOM_PASS` | Login password | open access |
| `AXIOM_ORG_NAME` | Org name on PDF | AXIOM Energy Audit Platform |
| `AXIOM_TAGLINE` | PDF footer tagline | Powered by Claude AI \| ISO 50002 |
| `AXIOM_COLOR` | Accent colour (hex) | #1A3C5E |

---

## 📐 Key Formulas

### Carbon-Adjusted Payback
```
Adj Payback = CapEx / (EUR savings/yr + (kgCO₂e/yr ÷ 1000) × €65)
```

### Weather Normalisation (HDD base 18°C)
```
Norm kWh = Base Load + Weather Load × (HDD_standard / HDD_actual)
Where: Base Load = 40% of consumption, Weather Load = 60%
```

### ESCO NPV
```
NPV = Σ(t=1 to N) [ CF_t / (1+r)^t ] - CapEx
```

---

## 🗺️ Roadmap

- [ ] Step 26 — IoT/CSV meter time-series upload + anomaly detection
- [ ] Step 27 — English / French language toggle
- [ ] Step 28 — REST API endpoint for programmatic audits
- [ ] Step 29 — Scope 1/2/3 GHG inventory module
- [ ] Step 30 — SBTi pathway calculator

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 🙏 Acknowledgements

- [Anthropic Claude](https://anthropic.com) — AI narrative generation
- [Open-Meteo](https://open-meteo.com) — Free weather archive API
- [ADEME](https://data.ademe.fr) — French energy benchmark data
- [Hugging Face](https://huggingface.co) — Free GPU hosting

---

*Built with AXIOM — the AI energy audit platform. ISO 50002:2014 | EU EED 2023/1791*
