import streamlit as st
import pandas as pd
import pickle
import numpy as np
from src.shap import shap_plot


@st.cache_resource
def load_model():
    with open("models/xgb_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("Wilson Disease Diagnoser")
st.write("Introduceți analizele pentru evaluare:")

cerulo = st.number_input("Ceruloplasmin Level", 0.0, 60.0, 10.0)
copper_serum = st.number_input("Copper in Blood Serum", 0.0, 1000.0, 100.0)
free_copper = st.number_input("Free Copper in Blood Serum", 0.0, 200.0, 20.0)
copper_urine = st.number_input("Copper in Urine", 0.0, 300.0, 50.0)

alt = st.number_input("ALT", 0.0, 1000.0, 30.0)
ast = st.number_input("AST", 0.0, 1000.0, 30.0)
total_bil = st.number_input("Total Bilirubin", 0.0, 20.0, 1.0)
albumin = st.number_input("Albumin", 0.0, 10.0, 4.0)
alp = st.number_input("Alkaline Phosphatase (ALP)", 0.0, 500.0, 100.0)

inr = st.number_input("Prothrombin Time / INR", 0.0, 10.0, 1.1)
ggt = st.number_input("Gamma-Glutamyl Transferase (GGT)", 0.0, 500.0, 50.0)

neuro = st.number_input("Neurological Symptoms Score", 0, 10, 0)
psych = st.number_input("Psychiatric Symptoms", 0, 1, 0)
cognitive = st.number_input("Cognitive Function Score", 0, 100, 80)
family_history = st.number_input("Family History", 0, 1, 0)

bmi = st.number_input("BMI", 0.0, 60.0, 22.0)

sex = st.selectbox("Sex", ["Female", "Male"])
alcohol = st.selectbox("Alcohol Use", ["False", "True"])

# One-hot encoding exact ca în dataset
Sex_Female = 1 if sex == "Female" else 0
Sex_Male = 1 if sex == "Male" else 0
Alcohol_False = 1 if alcohol == "False" else 0
Alcohol_True = 1 if alcohol == "True" else 0

# Build dataframe EXACT în ordinea din model
input_df = pd.DataFrame([{
    "Ceruloplasmin Level": cerulo,
    "Copper in Blood Serum": copper_serum,
    "Free Copper in Blood Serum": free_copper,
    "Copper in Urine": copper_urine,
    "ALT": alt,
    "AST": ast,
    "Total Bilirubin": total_bil,
    "Albumin": albumin,
    "Alkaline Phosphatase (ALP)": alp,
    "Prothrombin Time / INR": inr,
    "Gamma-Glutamyl Transferase (GGT)": ggt,
    "Neurological Symptoms Score": neuro,
    "Psychiatric Symptoms": psych,
    "Cognitive Function Score": cognitive,
    "Family History": family_history,
    "BMI": bmi,
    "Sex_Female": Sex_Female,
    "Sex_Male": Sex_Male,
    "Alcohol Use_False": Alcohol_False,
    "Alcohol Use_True": Alcohol_True
}])

st.write("### Date introduse")
st.dataframe(input_df)



if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Rezultat:")
    if prediction == 1:
        st.error(f"❗ Probabilitate mare de Wilson Disease ({prob:.2f})")
    else:
        st.success(f"✔ Probabilitate scăzută de Wilson Disease ({prob:.2f})")


    st.subheader("Interpretabilitate Model (SHAP)")
    with st.spinner("Se calculează importanța factorilor..."):
        fig = shap_plot(model, input_df)
        st.pyplot(fig)