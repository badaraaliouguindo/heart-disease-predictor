import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- Configuration de la page ---
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

# --- Chargement du modèle ---
@st.cache_resource
def load_model():
    model         = joblib.load("app/model.pkl")
    scaler        = joblib.load("app/scaler.pkl")
    feature_names = joblib.load("app/feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# --- Titre ---
st.title("🫀 Heart Disease Predictor")
st.markdown("Entrez les informations du patient pour prédire le risque de maladie cardiaque.")
st.divider()

# --- Sidebar : inputs patient ---
st.sidebar.header("Informations du patient")

age      = st.sidebar.slider("Âge", 20, 80, 54)
sex      = st.sidebar.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")
cp       = st.sidebar.selectbox("Type de douleur thoracique (cp)",
                                 [0, 1, 2, 3],
                                 format_func=lambda x: ["Asymptomatique", "Angine atypique",
                                                         "Douleur non-angineuse", "Angine typique"][x])
trestbps = st.sidebar.slider("Pression artérielle au repos (mmHg)", 90, 200, 130)
chol     = st.sidebar.slider("Cholestérol (mg/dl)", 120, 570, 246)
fbs      = st.sidebar.selectbox("Glycémie à jeun > 120 mg/dl", [0, 1],
                                 format_func=lambda x: "Non" if x == 0 else "Oui")
restecg  = st.sidebar.selectbox("ECG au repos", [0, 1, 2],
                                 format_func=lambda x: ["Normal",
                                                         "Anomalie ST-T",
                                                         "Hypertrophie ventriculaire"][x])
thalach  = st.sidebar.slider("Fréquence cardiaque max (bpm)", 70, 205, 150)
exang    = st.sidebar.selectbox("Angine à l'effort", [0, 1],
                                 format_func=lambda x: "Non" if x == 0 else "Oui")
oldpeak  = st.sidebar.slider("Dépression ST (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
slope    = st.sidebar.selectbox("Pente segment ST", [0, 1, 2],
                                 format_func=lambda x: ["Descendante", "Plate", "Ascendante"][x])
ca       = st.sidebar.slider("Nombre de vaisseaux majeurs (ca)", 0, 4, 0)
thal     = st.sidebar.selectbox("Thalassémie", [0, 1, 2, 3],
                                 format_func=lambda x: ["Inconnu", "Normal",
                                                         "Défaut fixe", "Défaut réversible"][x])

# --- Construction du vecteur patient ---
def build_input(age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal):
    base = {
        'age': age, 'sex': sex, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'ca': ca,
        'cp_0': 0, 'cp_1': 0, 'cp_2': 0, 'cp_3': 0,
        'restecg_0': 0, 'restecg_1': 0, 'restecg_2': 0,
        'slope_0': 0, 'slope_1': 0, 'slope_2': 0,
        'thal_0': 0, 'thal_1': 0, 'thal_2': 0, 'thal_3': 0,
    }
    base[f'cp_{cp}']      = 1
    base[f'restecg_{restecg}'] = 1
    base[f'slope_{slope}']     = 1
    base[f'thal_{thal}']       = 1

    df = pd.DataFrame([base])[feature_names]
    return df

input_df = build_input(age, sex, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak, slope, ca, thal)

# --- Prédiction ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Résultat de la prédiction")

    input_scaled = scaler.transform(input_df)
    proba        = model.predict_proba(input_scaled)[0][1]
    prediction   = int(proba >= 0.5)

    if prediction == 1:
        st.error(f"⚠️ Risque élevé de maladie cardiaque")
    else:
        st.success(f"Faible risque de maladie cardiaque")

    st.metric("Probabilité de maladie", f"{proba*100:.1f}%")
    st.progress(float(proba))

    st.divider()
    st.subheader("Données du patient")
    st.dataframe(input_df.T.rename(columns={0: "Valeur"}), use_container_width=True)

with col2:
    st.subheader("Pourquoi cette prédiction ? (SHAP)")

    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
    explainer       = shap.TreeExplainer(model)
    shap_values     = explainer.shap_values(input_scaled_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap':    shap_values[0]
    }).sort_values('shap', key=abs, ascending=True).tail(10)

    colors = ['#D85A30' if v > 0 else '#5DCAA5' for v in shap_df['shap']]
    ax.barh(shap_df['feature'], shap_df['shap'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title("Top 10 features influençant la prédiction")
    ax.set_xlabel("Valeur SHAP (rouge = augmente le risque, vert = réduit le risque)")
    st.pyplot(fig)

# --- Footer ---
st.divider()
