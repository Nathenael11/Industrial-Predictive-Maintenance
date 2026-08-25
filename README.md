# Industrial Predictive Maintenance System

![Elevvo AI](static/img/logo.svg)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-brightgreen)](https://industrial-predictive-maintenance.onrender.com)
[![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render&logoColor=white)](https://industrial-predictive-maintenance.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Production-Ready Predictive Maintenance System utilizing the UCI AI4I 2020 Dataset.**  
> Built with XGBoost, Flask, Docker, and an FDR-Optimized decision engine to minimize costly false alarms in industrial manufacturing.

---

## 🌐 Live Demo

| | |
| :--- | :--- |
| **URL** | [https://industrial-predictive-maintenance.onrender.com](https://industrial-predictive-maintenance1.onrender.com) |
| **Deployment Platform** | Render.com |
| **Status** | ✅ Live |

> **Note:** The app is hosted on Render's free tier, so the instance may spin down after periods of inactivity — the first request after idling can take a few seconds to wake it back up.

---

## 📌 Executive Overview

In modern smart manufacturing, unexpected machine breakdowns cause severe operational disruptions. However, **false alarms (false discoveries)** are equally detrimental—dispatching maintenance technicians to inspect functional equipment wastes hundreds of labor hours and causes unnecessary line stoppages.

This project delivers an end-to-end Machine Learning solution designed specifically to minimize the **False Discovery Rate (FDR)**:

$$\text{FDR} = \frac{\text{False Positives (FP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}} = 1 - \text{Precision}$$

### 🏆 Empirical Benchmark Results (AI4I 2020 Dataset)

| Metric | Default Threshold (0.50) | Optimal Threshold (0.76) | Impact / Improvement |
| :--- | :---: | :---: | :--- |
| **False Discovery Rate (FDR)** | **53.12%** | **31.15%** | 📉 **41.4% FDR Reduction** (Significantly fewer false alarms) |
| **Precision** | 46.88% | 68.85% | 📈 **+21.97% Improvement** |
| **Recall** | 88.24% | 82.35% | ✅ Sustained $\ge 70\%$ operational safety recall target |
| **F1-Score** | 0.6122 | 0.7500 | 📈 **+13.78% Overall Quality Boost** |

---

## 🏗 System Architecture

```
                                  +---------------------------------------+
                                  |    AI4I 2020 Dataset (UCI Repository) |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |  Feature Engineering & Leakage Filter |
                                  |  (Drop UDI, Prod ID, TWF, HDF, PWF)   |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |   XGBoost Classifier + ScalePosWeight |
                                  |   (AUC-PR Early Stopping 25 Rounds)    |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |  FDR Threshold Optimization (Th=0.76) |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |  Model Serialization (model_artifacts) |
                                  +---------------------------------------+
                                                      |
                                      +---------------+---------------+
                                      |                               |
                                      v                               v
                        +---------------------------+   +---------------------------+
                        |  Flask Web App (app.py)   |   |   Jupyter Notebook        |
                        |  - Single & Batch API     |   |   (predictive_maint.ipynb)|
                        |  - Dark Dashboard UI      |   +---------------------------+
                        +---------------------------+
                                      |
                                      v
                        +---------------------------+
                        |  Docker & Docker-Compose  |
                        |  (gunicorn WSGI Server)   |
                        +---------------------------+
```

---

## 📁 Repository Structure

```
predictive-maintenance/
├── predictive_maintenance.ipynb  # Complete 11-cell Jupyter ML Notebook
├── train_model.py                # Model training & artifact generation script
├── app.py                        # Production Flask Web Server & API
├── run.py                        # Entrypoint script for local development
├── requirements.txt              # Pinned Python dependencies
├── Dockerfile                    # Multi-stage security-hardened Docker container
├── docker-compose.yml            # Docker Compose setup
├── deploy.sh                     # Automated EC2 deployment bash script
├── README.md                     # Comprehensive project documentation
├── CHANGELOG.md                  # Release version history
├── LICENSE                       # MIT License
├── .gitignore                    # Python & Docker ignore rules
├── model_artifacts/
│   ├── model.pkl                 # Serialized XGBoost model
│   ├── scaler.pkl                # Standard Scaler object
│   └── metrics.json              # Optimal threshold & performance metadata
├── templates/
│   ├── index.html                # Single prediction dashboard with risk gauge
│   └── batch.html                # Drag-and-drop batch CSV processing view
├── static/
│   ├── css/style.css             # Dark theme design system (AML Shield aesthetic)
│   ├── js/script.js              # AJAX, SVG Gauge, table management & CSV export
│   └── img/logo.svg              # SVG Shield logo
├── tests/
│   └── test_app.py               # Pytest suite (100% pass rate)
└── .github/
    └── workflows/
        └── deploy.yml            # CI/CD test, build, and deployment pipeline
```

---

## 🚀 Quick Start Guide

### Option 1: Local Python Execution

1. **Clone & Navigate into directory**:
   ```bash
   git clone https://github.com/Nathenael11/Industrial-Predictive-Maintenance.git
   cd Industrial-Predictive-Maintenance
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train Model & Generate Artifacts**:
   ```bash
   python train_model.py
   ```

4. **Launch Flask Server**:
   ```bash
   python run.py
   ```
   Open your browser at `http://localhost:5000`.

---

### Option 2: Docker Container Execution

1. **Build and Run with Docker Compose**:
   ```bash
   docker-compose up --build -d
   ```

2. **Access Web App**:
   Navigate to `http://localhost:5000`.

3. **Check Container Health**:
   ```bash
   curl http://localhost:5000/health
   ```

---

### Option 3: Automated EC2 / Linux Server Deployment

Run the automated deployment script on any Linux/EC2 machine:

```bash
chmod +x deploy.sh
./deploy.sh
```

---

## 🧪 Running Unit Tests

Execute the comprehensive test suite with `pytest`:

```bash
pytest tests/test_app.py -v
```

---

## 🌐 API Documentation

### 1. Single Telemetry Risk Prediction
- **Endpoint**: `POST /predict`
- **Content-Type**: `application/json` or `application/x-www-form-urlencoded`
- **Payload Example**:
  ```json
  {
    "air_temp": 300.0,
    "process_temp": 310.0,
    "rotational_speed": 1500,
    "torque": 40.0,
    "tool_wear": 120,
    "product_type": "L"
  }
  ```
- **Response**:
  ```json
  {
    "risk_score": 12.4,
    "prediction": "No Failure",
    "is_failure": false,
    "confidence": 91.8,
    "threshold_used": 0.76,
    "recommended_action": "NORMAL: Machine telemetry operating within standard nominal parameters."
  }
  ```

### 2. Batch CSV Upload Prediction
- **Endpoint**: `POST /batch`
- **Payload**: Multipart form upload with key `file` containing `.csv`.

### 3. API Health Check
- **Endpoint**: `GET /health`
- **Response**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "optimal_threshold": 0.76,
    "service": "Elevvo Predictive Maintenance API",
    "version": "1.0.0"
  }
  ```

---

## 🛠 Technologies Used

- **Machine Learning**: Python 3.10+, XGBoost, Scikit-Learn, Pandas, NumPy, Joblib
- **Data Visualization**: Matplotlib, Seaborn
- **Web Framework**: Flask, Jinja2, Werkzeug, Gunicorn
- **Frontend**: Vanilla HTML5, Modern CSS3 (Dark Theme, Glassmorphism), SVG Animations, JavaScript ES6
- **DevOps & CI/CD**: Docker, Docker-Compose, Pytest, Bash, GitHub Actions
- **Deployment**: Render.com

---

## 🙏 Acknowledgments

- **[Elevvo Internship Program](https://elevvo.tech/)** — for the project brief and mentorship framework
- **[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)** — for the AI4I 2020 Predictive Maintenance Dataset
- **[Render.com](https://render.com/)** — for free and reliable application hosting

---

## 👤 Author

**Nathenael Ermias**

- 📧 Email: [nathnaelermias@gmail.com](mailto:nathnaelermias@gmail.com)
- 💻 GitHub: [@Nathenael11](https://github.com/Nathenael11)
- 🔗 LinkedIn: [nathenael-ermias](https://www.linkedin.com/in/nathenael-ermias)

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.
