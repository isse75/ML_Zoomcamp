import streamlit as st
import requests

# Get your EB API URL from secrets
API_URL = st.secrets["api"]["url"]

st.set_page_config(
    page_title="Heart Disease Predictor",
    layout="centered",
)

st.title("❤️ Heart Disease Risk Predictor")

# Sidebar or main inputs
st.header("Patient Data")
Age = st.number_input("Age", min_value=1, max_value=120, value=50)
Sex = st.selectbox("Sex", ["Male", "Female"])
ChestPainType = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
RestingBP = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
Cholesterol = st.number_input("Cholesterol", min_value=50, max_value=600, value=200)
FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
RestingECG = st.selectbox("Resting ECG Result", [0, 1, 2])
MaxHR = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
ExerciseAngina = st.selectbox("Exercise-Induced Angina", [0, 1])
Oldpeak = st.number_input("ST Depression (Oldpeak)", format="%.1f", value=1.0)
ST_Slope = st.selectbox("ST Slope", [0, 1, 2])
Ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
Thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Predict button
if st.button("Predict Heart Disease Risk"):
    payload = {
        "Age": Age,
        "Sex": 1 if Sex == "Male" else 0,
        "ChestPainType": ChestPainType,
        "RestingBP": RestingBP,
        "Cholesterol": Cholesterol,
        "FastingBS": FastingBS,
        "RestingECG": RestingECG,
        "MaxHR": MaxHR,
        "ExerciseAngina": ExerciseAngina,
        "Oldpeak": Oldpeak,
        "ST_Slope": ST_Slope,
        "Ca": Ca,
        "Thal": Thal
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json().get("heart_disease")
        if result:
            st.error("Prediction: High risk of heart disease ❤️")
        else:
            st.success("Prediction: Low risk of heart disease 👍")
    except Exception as e:
        st.error(f"API call failed: {e}")
