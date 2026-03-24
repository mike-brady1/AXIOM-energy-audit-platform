# AXIOM Energy Audit Platform

> Automated EU-compliant energy audits powered by Claude AI and ADEME open data.

[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-blue)](https://huggingface.co/spaces/Mr-MB/axiom-energy-audit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What It Does

AXIOM is an AI-powered automated energy audit platform that:

- **Benchmarks** any French tertiary building against 1M+ ADEME open data records
- **Identifies** Energy Conservation Measures (ECMs) with kWh, EUR, and kgCO2e savings
- **Generates** a professional PDF report (ASHRAE Level I, ISO 50002:2014)
- **Complies** with EU Energy Efficiency Directive 2023/1791

## Features

| Feature | Detail |
|---------|--------|
| Benchmarking | 9 building activity types, ADEME tertiary dataset |
| ECMs | BMS, LED, HVAC Heat Pump, Building Envelope |
| AI Narrative | Claude Sonnet — executive summary, plain text |
| PDF Report | ReportLab — profile table, EUI chart, ECM table, summary |
| Standards | ISO 50002:2014, ASHRAE Level I, EU EED 2023/1791 |

## Tech Stack

- **Frontend**: Gradio 4.x (Hugging Face Spaces)
- **AI**: Anthropic Claude claude-sonnet-4-5
- **PDF**: ReportLab
- **Charts**: Matplotlib
- **Data**: ADEME open data (data.ademe.fr)

## Quick Start

```bash
git clone https://github.com/mike-brady1/AXIOM-energy-audit-platform
cd AXIOM-energy-audit-platform
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key
python app.py
```

