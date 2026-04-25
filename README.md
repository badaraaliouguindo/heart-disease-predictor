#  Heart Disease Predictor — ML with Explainability

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

> Predicting the risk of heart disease using an interpretable machine learning model — with real-time SHAP explanations for each individual prediction.

**[ Live Demo](https://heart-disease-predictor-vxbxp54gfyv3kjsvxha7io.streamlit.app/)** | **[ EDA Notebook](notebooks/01_EDA_Heart_Disease.ipynb)** | **[ Modeling Notebook](notebooks/02_Modeling.ipynb)** | **[ SHAP Notebook](notebooks/03_SHAP_Analysis.ipynb)**

---

## Overview

This project builds an end-to-end machine learning pipeline to predict whether a patient is at risk of heart disease. Beyond accuracy, the focus is on **model explainability** using SHAP values — because in healthcare, understanding *why* a model makes a prediction is as important as the prediction itself.

---

## Demo

![App Screenshot](https://via.placeholder.com/900x450?text=Heart+Disease+Predictor+App)

> Adjust patient parameters in the sidebar → get an instant prediction + SHAP explanation

---

## Results

| Metric | Score |
|--------|-------|
| AUC-ROC (test set) | **0.857** |
| AUC-ROC (5-fold CV) | **0.884 ± 0.035** |
| Accuracy | **77%** |
| F1-score (disease class) | **0.80** |

The low standard deviation across folds (0.035) confirms the model is stable and generalizes well.

---

## Key Findings

Through SHAP analysis, the top predictive features are:

- **thal_2** — Thalassemia type (fixed defect) is the strongest predictor
- **cp_0** — Absence of chest pain reduces disease risk
- **ca** — Number of major vessels: more blocked vessels → higher risk
- **chol** — Cholesterol level has a significant impact
- **oldpeak** — ST depression induced by exercise

These findings are clinically consistent with established cardiology literature.

>  **Clinical note**: The model produced 5 false negatives (sick patients classified as healthy) on the test set. In a real medical setting, this threshold would need to be adjusted to prioritize recall over precision.

---

## Project Structure

heart-disease-predictor/
├── notebooks/
│   ├── 01_EDA_Heart_Disease.ipynb      # Exploratory Data Analysis
│   ├── 02_Modeling.ipynb               # Model training & evaluation
│   └── 03_SHAP_Analysis.ipynb          # Explainability analysis
├── app/
│   ├── streamlit_app.py                # Interactive web application
│   ├── model.pkl                       # Trained XGBoost model
│   ├── scaler.pkl                      # StandardScaler
│   └── feature_names.pkl              # Feature names
├── requirements.txt
└── README.md

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| ML Model | XGBoost |
| Explainability | SHAP |
| Data Processing | pandas, scikit-learn |
| Visualization | matplotlib, seaborn |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |

---

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/badaraaliouguindo/heart-disease-predictor.git
cd heart-disease-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/streamlit_app.py
```

---

## Dataset

- **Source**: [Heart Disease UCI — Kaggle](https://www.kaggle.com/datasets/rohanharode07/heart-disease-uci)
- **Size**: 303 patients, 14 features
- **Target**: Binary (0 = no disease, 1 = heart disease)
- **Class balance**: 54.5% positive / 45.5% negative

---

## Author

**Badara Aliou Guindo**
Master's student in Data Science & Artificial Intelligence

[![GitHub](https://img.shields.io/badge/GitHub-badaraaliouguindo-black)](https://github.com/badaraaliouguindo)

---
