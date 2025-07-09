import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the trained model
with open("transplant_model.pkl", "rb") as f:
    model = pickle.load(f)

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

# Feature encoding (should match training encoding)
def label_encode(value, choices):
    return choices.index(value)

if st.button("Predict"):
    # Manual encoding based on training
    encoded_input = {
        "Gender_Patient": 0 if gender_patient == "Male" else 1,
        "Gender_Donor": 0 if gender_donor == "Male" else 1,
        "Age_Difference": abs(age_patient - age_donor),
        "Organ": label_encode(organ, ["Kidney", "Liver", "Heart", "Lung"]),
        "HLA Match": label_encode(hla_match, ["Low", "Moderate", "High"]),
        "Location": label_encode(location, ["Same City", "Same Country", "International"]),
        "Donor Blood Group": label_encode(donor_blood_group, ["A", "B", "AB", "O"]),
        "Patient Blood Group": label_encode(patient_blood_group, ["A", "B", "AB", "O"]),
    }

    input_df = pd.DataFrame([encoded_input])

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Output
    if prediction == 1:
        st.success(f"✅ Eligible for transplant! (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ Not eligible for transplant. (Confidence: {probability:.2f})")
