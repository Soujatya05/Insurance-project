import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Insurance Predictor", page_icon="💰")

st.title("💰 Insurance Cost Predictor")

from pathlib import Path

# Load model and scaler from this script's directory
base_path = Path(__file__).resolve().parent
model = pickle.load(open(base_path / "model.pkl", "rb"))
scaler = pickle.load(open(base_path / "scaler.pkl", "rb"))

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100)
    bmi = st.number_input("BMI", 10.0, 50.0)
    children = st.number_input("Children", 0, 10)

with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Encoding
is_female = 1 if sex == "female" else 0
is_smoker = 1 if smoker == "yes" else 0
region_southeast = 1 if region == "southeast" else 0
region_northwest = 1 if region == "northwest" else 0
bmi_category_Obese = 1 if bmi >= 30.0 else 0

if st.button("Predict"):
    scaled = scaler.transform([[age, bmi, children]])
    final_input = np.array([[
        scaled[0][0],
        is_female,
        scaled[0][1],
        scaled[0][2],
        is_smoker,
        region_southeast,
        bmi_category_Obese,
        region_northwest,
    ]])
    prediction = model.predict(final_input)

    st.success(f"💸 Cost: ₹ {prediction[0]:,.2f}")