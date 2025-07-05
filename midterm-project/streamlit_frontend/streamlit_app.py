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

# Descriptive mappings for user-friendly labels
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

# Collect inputs
Age = st.number_input("Age", min_value=1, max_value=120, value=50)
Sex = st.selectbox("Sex", ["Male", "Female"])
ChestPainType_label = st.selectbox("Chest Pain Type", list(cp_map.keys()))
RestingBP = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
Cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=600, value=200)
FastingBS_label = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
RestingECG_label = st.selectbox("Resting ECG Result", list(ecg_map.keys()))
MaxHR = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
ExerciseAngina_label = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
Oldpeak = st.number_input("ST Depression (Oldpeak)", format="%.1f", value=1.0)
ST_Slope = st.selectbox("ST Slope", [0, 1, 2])
Ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
Thal_label = st.selectbox("Thalassemia", list(thal_map.keys()))

# Trigger prediction
if st.button("Predict Heart Disease Risk"):
    payload = {
        "age": Age,
        "sex": 1 if Sex == "Male" else 0,
        "chest_pain_type": cp_map[ChestPainType_label],
        "resting_bp": RestingBP,
        "cholesterol": Cholesterol,
        "fasting_blood_sugar": True if FastingBS_label == "Yes" else False,
        "resting_ecg": ecg_map[RestingECG_label],
        "max_hr_achieved": MaxHR,
        "exercise_angina": True if ExerciseAngina_label == "Yes" else False,
        "st_depression": Oldpeak,
        "st_slope": ST_Slope,
        "no_vessels_fluoroscopy": Ca,
        "thal_result": thal_map[Thal_label]
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
