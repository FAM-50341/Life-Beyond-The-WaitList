import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
try:
    with open("transplant_model.pkl", "rb") as f:
        model = joblib.load(f)
except Exception as e:
    st.error(f"❌ Failed to load the model: {e}")
    st.stop()

st.set_page_config(page_title="🧬 Transplant Eligibility Predictor", layout="centered")
st.title("🧬 Organ Transplant Eligibility Predictor")
st.markdown("Provide the necessary details below to check eligibility prediction.")

# Input Fields
gender_patient = st.selectbox("Gender (Patient)", ["Male", "Female"])
gender_donor = st.selectbox("Gender (Donor)", ["Male", "Female"])

age_patient = st.slider("Patient Age", 0, 100, 35)
age_donor = st.slider("Donor Age", 0, 100, 40)

organ = st.selectbox("Organ", ["Kidney", "Liver", "Heart", "Lung"])
hla_match = st.selectbox("HLA Match", ["Low", "Moderate", "High"])
location = st.selectbox("Location", ["Same City", "Same Country", "International"])
donor_blood_group = st.selectbox("Donor Blood Group", ["A", "B", "AB", "O"])
patient_blood_group = st.selectbox("Patient Blood Group", ["A", "B", "AB", "O"])

# Encoding helper
def label_encode(value, choices):
    return choices.index(value)

# Prediction
if st.button("Predict"):
    raw_input = {
        "Gender_Patient": 0 if gender_patient == "Male" else 1,
        "Gender_Donor": 0 if gender_donor == "Male" else 1,
        "Age_Difference": abs(age_patient - age_donor),
        "Organ": organ,
        "HLA Match": hla_match,
        "Location": location,
        "Donor Blood Group": donor_blood_group,
        "Patient Blood Group": patient_blood_group,
    }

    input_df = pd.DataFrame([raw_input])

    # 🔥 One-hot encode categorical columns (must match training code)
    categorical_cols = ["Organ", "HLA Match", "Location", "Donor Blood Group", "Patient Blood Group"]
    input_df = pd.get_dummies(input_df, columns=categorical_cols)

    # 🧠 Align with training model's expected features
    missing_cols = set(model.feature_names_in_) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    # ✅ Predict
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        print(prediction, probability)
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # 🎉 Show result
    if prediction == 1:
        st.success(f"✅ Eligible for transplant! (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ Not eligible for transplant. (Confidence: {probability:.2f})")
