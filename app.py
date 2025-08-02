import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les objets sauvegardés
model = joblib.load("best_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Configuration de la page
st.set_page_config(page_title="Prédiction du Diabète", layout="wide")
st.title("\U0001F489 Prédiction du Diabète avec le Modèle Entraîné")

# Sidebar pour les entrées utilisateur
st.sidebar.header("Entrées Patient")
pregnancies = st.sidebar.slider("Nombre de grossesses", 0, 20, 1)
glucose = st.sidebar.slider("Taux de glucose", 40, 200, 100)
blood_pressure = st.sidebar.slider("Pression artérielle", 20, 140, 70)
skin_thickness = st.sidebar.slider("Epaisseur de peau (mm)", 0, 100, 20)
insulin = st.sidebar.slider("Insuline (mu U/ml)", 0, 900, 80)
bmi = st.sidebar.slider("IMC (BMI)", 10.0, 70.0, 25.0)
dpf = st.sidebar.slider("Hérédité du diabète (DPF)", 0.0, 3.0, 0.5) 
age = st.sidebar.slider("Âge", 15, 100, 33)

# Encapsuler dans un DataFrame
data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})

# Affichage des données utilisateur
st.subheader("Données du Patient")
st.dataframe(data)

# Visualisations pertinentes (histogrammes de la base de données)
st.subheader("Distribution de quelques variables cliniques")
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(data=data, x="Glucose", bins=20, ax=axs[0], color='skyblue')
axs[0].set_title("Taux de Glucose")
sns.histplot(data=data, x="BMI", bins=20, ax=axs[1], color='salmon')
axs[1].set_title("Indice de Masse Corporelle (IMC)")
st.pyplot(fig)

# Bouton de prédiction
if st.button("Lancer la Prédiction"):
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)[0]
    proba = model.predict_proba(scaled_data)[0][1]  # Probabilité d'être diabétique

    st.subheader("Résultat de la Prédiction")
    if prediction == 1:
        st.error(f"Le patient est probablement diabétique (π = {proba:.2f})")
    else:
        st.success(f"Le patient est probablement non diabétique (π = {proba:.2f})")

    st.subheader("Conseil Personnalisé")
    if proba < 0.3:
        st.info("\u2705 Le risque est faible. Pensez à maintenir une bonne hygiène de vie.")
    elif 0.3 <= proba < 0.7:
        st.warning("\u26A0\ufe0f Risque modéré. Un contrôle médical est recommandé.")
    else:
        st.error("\u2620\ufe0f Risque élevé. Consultez rapidement un professionnel de santé.")
