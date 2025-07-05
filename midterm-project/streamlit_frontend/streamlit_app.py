import streamlit as st
import requests

# Get your EB API URL from secrets
API_URL = st.secrets["api"]["url"]

st.set_page_config(
    page_title="Heart Disease Predictor",
    layout="centered",
)

st.title("❤️ Heart Disease Risk Predictor")
st.header("Patient Data")

# Descriptive mappings for categorical variables
cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

ecg_map = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

thal_map = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain_type_label = st.selectbox("Chest Pain Type", list(cp_map.keys()))
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=600, value=200)
fasting_blood_sugar_label = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
resting_ecg_label = st.selectbox("Resting ECG Result", list(ecg_map.keys()))
max_hr_achieved = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
exercise_angina_label = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
st_depression = st.number_input("ST Depression (Oldpeak)", format="%.1f", value=1.0)
st_slope = st.selectbox("ST Slope", [0, 1, 2])
no_vessels_fluoroscopy = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal_result_label = st.selectbox("Thalassemia", list(thal_map.keys()))

# Button to make prediction
if st.button("Predict Heart Disease Risk"):
    payload = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "chest_pain_type": cp_map[chest_pain_type_label],
        "resting_bp": resting_bp,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": True if fasting_blood_sugar_label == "Yes" else False,
        "resting_ecg": ecg_map[resting_ecg_label],
        "max_hr_achieved": max_hr_achieved,
        "exercise_angina": True if exercise_angina_label == "Yes" else False,
        "st_depression": st_depression,
        "st_slope": st_slope,
        "no_vessels_fluoroscopy": no_vessels_fluoroscopy,
        "thal_result": thal_map[thal_result_label]
    }

    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json().get("heart_disease")

        if result:
            st.error("Prediction: High risk of heart disease")
        else:
            st.success("Prediction: Low risk of heart disease")

    except Exception as e:
        st.error(f"API call failed: {e}")
