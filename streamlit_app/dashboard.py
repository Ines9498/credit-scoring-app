# dashboard.py

import streamlit as st
import pandas as pd
import requests
import shap
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Prédiction Crédit Client", layout="centered")
st.title("📊 Prédiction de Solvabilité Client")

# Charger le modèle localement pour SHAP
@st.cache_resource
def load_model():
    with open("lightgbm_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Permettre à l'utilisateur d'uploader un fichier CSV
uploaded_file = st.file_uploader("📤 Charger un fichier CSV avec les données clients :", type=["csv"])

if uploaded_file is not None:
    clients_df = pd.read_csv(uploaded_file)

    # Choisir un client
    client_ids = clients_df["client_id"].tolist()
    selected_id = st.selectbox("Sélectionnez un client :", client_ids)

    client_data = clients_df[clients_df["client_id"] == selected_id].drop(columns=["client_id"])

    st.subheader("🧾 Données du client")
    st.write(client_data)

    # Appel à l'API Render
    if st.button("🔍 Prédire"):  
        payload = {"data": client_data.iloc[0].to_dict()}
        API_URL = "https://credit-scoring-app-jbzr.onrender.com/predict"
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            pred = result["prediction"]
            proba = result["proba_good_client"]

            st.success(f"✅ Prédiction : {'Bon client' if pred == 1 else 'Mauvais client'}")
            st.info(f"Probabilité d'être un bon client : {proba * 100:.2f}%")

            # Affichage SHAP
            st.subheader("📈 Interprétation locale (SHAP)")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(client_data)

            values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
            base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

            fig, ax = plt.subplots(figsize=(10, 3))
            shap.plots.waterfall(shap.Explanation(
                values=values,
                base_values=base_value,
                data=client_data.iloc[0]
            ), show=False)
            st.pyplot(fig)

        else:
            st.error(f"Erreur lors de la prédiction : {response.status_code}")
else:
    st.warning("Veuillez charger un fichier CSV pour commencer.")
