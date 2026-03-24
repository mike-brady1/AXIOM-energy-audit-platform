# AXIOM: AI Automated Energy Audit Platform

EU-compliant SaaS for buildings/industry audits (EED, ISO 50002). Automates data ingestion → AI analysis → reports.

## Quickstart (Colab)
1. Open [01_data_ingestion.ipynb](notebooks/01_data_ingestion.ipynb)
2. Upload sample PDF → Run → Get Parquet outputs.

## Architecture
5 Layers: Ingestion → Processing → AI → Compliance → Reports [Details](docs/AXIOM_prompt.md)

## Tech Stack
- Python 3.10, pandas, scikit-learn, Claude API
- GitHub Actions CI/CD

## License
MIT
