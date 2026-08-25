# Changelog

All notable changes to the **Elevvo Predictive Maintenance** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-24

### Added
- **Jupyter Pipeline**: Completed `predictive_maintenance.ipynb` featuring EDA, leakage mitigation, XGBoost classification, decision threshold optimization, and submission exports.
- **Model Training Engine**: Created standalone `train_model.py` generating `model.pkl`, `scaler.pkl`, and `metrics.json`.
- **False Discovery Rate (FDR) Optimization**: Implemented threshold tuning (Optimal = 0.76) that dropped FDR from 53.12% to 31.15% while sustaining Recall $\ge 70\%$.
- **Flask Web Server (`app.py`)**: Real-time `/predict` POST endpoint, session prediction history, batch upload `/batch` endpoint, CSV template downloader `/download_template`, and `/health` API.
- **Dark-Themed UI (`templates/index.html` & `batch.html`)**: Sleek dark navy/teal UI matching AML Shield quality with SVG risk gauge animations, KPI cards, interactive tables, and drag-and-drop batch upload.
- **Unit Testing**: Pytest suite `tests/test_app.py` achieving 100% test pass rate across data preprocessing, inference, and HTTP routes.
- **Docker Containerization**: Multi-stage `Dockerfile` and `docker-compose.yml` configured for port 5000 with gunicorn WSGI server.
- **CI/CD & Deployment**: Automated `deploy.sh` script for Linux/EC2 environments and GitHub Actions `.github/workflows/deploy.yml` pipeline.
