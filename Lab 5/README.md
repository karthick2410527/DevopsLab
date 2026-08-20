# 🚀 MLOps CI/CD Pipeline for California Housing Price Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![GitHub Actions](https://img.shields.io/badge/GitHub-Actions-success)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Deployed-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end **MLOps CI/CD pipeline** that automatically prepares data, trains a machine learning model, evaluates its performance, runs unit tests, and deploys the trained model to **Hugging Face Hub** using **GitHub Actions**.

---

# 📌 Project Overview

This project demonstrates how Machine Learning models can be automatically built and deployed using modern MLOps practices.

Whenever code is pushed to the **main** branch:

- ✅ Unit Tests execute
- ✅ Dataset is prepared
- ✅ Model is trained
- ✅ Performance is evaluated
- ✅ Quality Gate checks model performance
- ✅ If the model passes evaluation, it is automatically deployed to Hugging Face Hub.

---

# 🏗️ Project Architecture

```
Developer Pushes Code
          │
          ▼
 GitHub Repository
          │
          ▼
 GitHub Actions
          │
          ▼
 Prepare Data
          │
          ▼
 Train Model
          │
          ▼
 Evaluate Model
          │
          ▼
 Quality Gate
          │
     Pass / Fail
          │
          ▼
 Deploy to Hugging Face Hub
```

---

# 📂 Project Structure

```
Lab 5
│
├── .github/
│   └── workflows/
│       └── ci-cd.yaml
│
├── src/
│   ├── prepare.py
│   ├── train.py
│   ├── evaluate.py
│   └── register.py
│
├── tests/
│   └── test_pipeline.py
│
├── model/
│   ├── model.joblib
│   └── features.json
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── params.yaml
├── requirements.txt
├── metrics.json
└── README.md
```

---

# ⚙️ Tech Stack

- Python 3.11
- Scikit-Learn
- Pandas
- NumPy
- Joblib
- PyYAML
- PyTest
- GitHub Actions
- Hugging Face Hub

---

# 🔄 CI/CD Pipeline

## Stage 1 — Data Preparation

- Loads the California Housing Dataset
- Splits data into Train/Test datasets
- Stores datasets as CSV files

Output

```
data/train.csv
data/test.csv
```

---

## Stage 2 — Model Training

- Loads training data
- Trains a Random Forest Regressor
- Saves trained model
- Saves feature names

Output

```
model/model.joblib
model/features.json
```

---

## Stage 3 — Model Evaluation

Calculates:

- MAE
- RMSE
- R² Score

If the model does not satisfy the quality threshold defined in `params.yaml`, the pipeline automatically stops.

---

## Stage 4 — Deployment

When evaluation succeeds:

- Generates Model Card
- Creates Hugging Face Repository (if needed)
- Uploads

```
model.joblib
features.json
README.md
```

Deployment is fully automated through GitHub Actions.

---



# ✅ Unit Testing

Implemented using **PyTest**.

Tests include:

- Dataset loading
- Data preparation
- Pipeline execution

Run locally:

```bash
pytest tests/
```

---

# 🚀 GitHub Actions Workflow

The workflow performs the following automatically:

```
Push to Main
      │
      ▼
Run Unit Tests
      │
      ▼
Prepare Dataset
      │
      ▼
Train Model
      │
      ▼
Evaluate Model
      │
      ▼
Quality Gate
      │
      ▼
Deploy to Hugging Face
```

---

# 📊 Model Performance

| Metric | Value |
|---------|------:|
| MAE | 0.3658 |
| RMSE | 0.5434 |
| R² Score | 0.7746 |

---

# 🌐 Hugging Face Deployment

The trained model is automatically deployed to Hugging Face Hub.

Repository:

```
https://huggingface.co/KanishkaRajesh/california-housing-regression
```

---

# 📸 Deployment Screenshot

<img width="1114" height="600" alt="image" src="https://github.com/user-attachments/assets/d5d2b157-ab60-4b5d-8656-76fbaeffc97b" />


# ▶️ Running Locally

Install dependencies

```bash
pip install -r requirements.txt
```

Prepare dataset

```bash
python src/prepare.py
```

Train model

```bash
python src/train.py
```

Evaluate model

```bash
python src/evaluate.py
```

Run tests

```bash
pytest tests/
```

---

# 💡 Key MLOps Concepts Demonstrated

- Continuous Integration (CI)
- Continuous Deployment (CD)
- Automated Testing
- Model Evaluation
- Quality Gates
- Artifact Generation
- Automated Model Deployment
- GitHub Secrets
- Hugging Face Hub Integration
- Reproducible ML Pipelines

---

# 📈 Future Improvements

- MLflow Experiment Tracking
- Docker Containerization
- Kubernetes Deployment
- DVC for Dataset Versioning
- Model Monitoring
- FastAPI Inference API
- Prometheus & Grafana Monitoring
- Scheduled Retraining

---

# 👨‍💻 Author

**Kanishka Rajesh**

