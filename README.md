# Projects_2025
All My Projects In 2025

Projects In The Order:
1. Store Closure

   # Store Closure Prediction Across Multi-Store Retail

A scalable ML pipeline (PySpark, Databricks-ready) to forecast store closures using sales, weather, and calendar data, generalized for multiple retail entities.

---

## Features
- End-to-end pipeline: Data prep → Feature engineering → Leakage-safe training → Evaluation
- Synthetic/demo data for reproducibility
- Proactive closure alerts for operational excellence

---

## How to Run

1. Clone repo: `git clone ...`
2. (Optional) Create/activate a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Open notebooks or run scripts in `src/`

---

## Folders

- `data/`: Demo synthetic CSVs
- `notebooks/`: Jupyter Notebooks (PySpark code)
- `src/`: Modular Python scripts for production

---

## Quickstart (Jupyter/Databricks)

```python
# 1. Generate or load synthetic data
# 2. Prepare and join dataframes
# 3. Feature engineering (rolling stats, lags, weather categoricals)
# 4. Split train/test (to avoid leakage)
# 5. Train ML model (RandomForestClassifier)
# 6. Evaluate & interpret results
exit
```

Project 2: Daily Prod Sales Affected By Weather Data
# 🛒 Multi-Store Sales Forecasting (Weather & Holiday-Aware)

This project demonstrates an end-to-end pipeline to forecast **daily product sales** across multiple store locations using PySpark. It incorporates features like holidays, weekends, and seasonal trends to improve prediction accuracy, even in the absence of real-world data.

## 📌 Objective
Predict daily sales for each store across the U.S. using a global model that generalizes across:
- Locations (businessEntityId)
- Product categories
- Time periods (holidays, weekends, seasons)

---

## 🔧 Tools & Technologies

- **Apache Spark (PySpark)**: Scalable distributed processing
- **Databricks / Spark Notebooks**: Development and data exploration
- **MLlib**: Machine Learning with Random Forest, XGBoost
- **Pandas** & **matplotlib/seaborn**: Local testing and visualization
- **Scikit-learn GridSearchCV (conceptual)**: Hyperparameter tuning
- **Git & GitHub**: Version control

---

## 📊 Features Engineered

- **Temporal**: Day of Week, Month, IsHoliday, IsWeekend, Rolling Averages
- **Lag Features**: 1-day, 7-day lags of sales
- **Store-Level**: businessEntityId (store ID), Product ID, etc.
- **Weather-aware model** (optional): Precipitation, Temperature, Humidity, etc.

---

## 🚀 ML Pipeline Steps

1. **Mock Data Generation** (for offline testing)
2. **Feature Engineering**
3. **Model Building** using RandomForestRegressor
4. **Evaluation** using RMSE, MAE, and MAPE
5. **Hyperparameter Tuning** with Grid Search (planned extension)
6. **Deployment-Ready Pipeline Structure**

---

## 🧪 Model Performance (Sample Results)

| Metric | Value |
|--------|-------|
| RMSE   | 105.31 |
| MAE    | 27.47 |
| MAPE   | 103.00% |

*MAPE can be optimized further with model tuning and refined seasonal adjustments.*

---

## 📁 Structure

Project 3: Sentiment Categorization w/ NLP

Sentiment Categorization & Retail Category Mapping

Table of Contents
	1.	Project Overview
	2.	Architecture
	3.	Prerequisites
	4.	Setup & Installation
	5.	Data Preparation (Bronze)
	6.	Feature Engineering (Silver)
	7.	Model Training & Registration
	8.	Inference & Gold Table
	9.	MLOps Pipeline (CI/CD)
	10.	Directory Structure
	11.	Next Steps

⸻

Project Overview

This end-to-end project ingests social media comments, performs sentiment and semantic categorization using SBERT and GPT-3.5, enriches with metadata (store weather, departments), trains a classification model, and deploys it on Azure Databricks with MLflow and a CI/CD pipeline.

Architecture

Raw Data (unified_posts_comments_testing)
        ↓ Bronze Cleaning & Deduplication
ml_bronze.comments_clean → pandas → text embeddings & HTTP calls
        ↓ Silver Feature Engineering & Enrichment
ml_silver.comments_features → model training (MLflow) → registered model
        ↓ Inference Pipeline
gold_dev.sentiment_predictions → downstream reporting

Additional GPT-based mapping for retail categories and store departments writes to gold_dev.sentiment_classify.

Prerequisites
	•	Azure Databricks workspace with attached cluster
	•	Access to OpenAI / Azure OpenAI (API key & deployment)
	•	MLflow workspace or Databricks-managed MLflow
	•	Tables: gold_dev.unified_posts_comments_testing, bronze.ovvi_department
	•	Databricks CLI or Git integration for CI/CD

Setup & Installation
	1.	Clone the repository into Databricks Repos.
	2.	Install required libraries on your cluster:

%pip install sentence-transformers requests mlflow


	3.	Configure environment variables or notebook-scoped variables for API keys and endpoints.

Data Preparation (Bronze)
	•	Notebook: 01_data_prep
	•	Reads gold_dev.unified_posts_comments_testing, cleans nulls, drops duplicates on comment_id and comment_text.
	•	Writes cleaned table to ml_bronze.comments_clean.

Feature Engineering (Silver)
	•	Notebook: 02_feature_engineering
	•	SBERT semantic categorization with all-MiniLM-L6-v2.
	•	GPT-3.5 mapping for retail categories and departments via REST API.
	•	Persists enriched table to ml_silver.comments_features.

Model Training & Registration
	•	Notebook: 03_train_register
	•	Loads silver table, extracts embeddings, trains a logistic regression classifier.
	•	Logs experiment and metrics to MLflow.
	•	Registers the model as SentimentCatModel with versioning.

Inference & Gold Table
	•	Notebook: 04_inference
	•	Loads registered model from MLflow model registry.
	•	Scores new comments, writes predictions to gold_dev.sentiment_predictions (or gold_dev.sentiment_classify).

MLOps Pipeline (CI/CD)
	•	Azure DevOps (or GitHub Actions) YAML pipelines to:
	1.	Build & Test: import notebooks, run 01_data_prep and 02_feature_engineering.
	2.	Train & Register: execute 03_train_register.
	3.	Deploy: configure a scheduled job to execute 04_inference daily.
	•	Example pipeline files in ci/ directory (config_test_run.json, config_train_run.json, config_inference_job.json).

Directory Structure

/Repos/YourProject
├── 01_data_prep.ipynb
├── 02_feature_engineering.ipynb
├── 03_train_register.ipynb
├── 04_inference.ipynb
├── ci/
│   ├── azure-pipelines.yml
│   ├── config_test_run.json
│   ├── config_train_run.json
│   └── config_inference_job.json
└── README.md

Next Steps
	•	Implement monitoring on prediction drift and data quality.
	•	Add automated retraining triggers based on performance metrics.
	•	Extend the classification model to multi-label or hierarchical categories.
	•	Integrate real-time streaming inference using Databricks Structured Streaming.

⸻

Prepared by Data Engineering & ML Team
