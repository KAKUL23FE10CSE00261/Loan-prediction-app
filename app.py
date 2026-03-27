import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Loan AI System", layout="wide")

# =========================
# LOAD FILES
# =========================
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("loan_prediction.csv")

# =========================
# HEADER
# =========================
st.markdown("<h1 style='text-align:center;'>🏦 Loan Prediction Dashboard</h1>", unsafe_allow_html=True)

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("📌 Applicant Details")

Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Married = st.sidebar.selectbox("Married", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
Property_Area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
Credit_History = st.sidebar.selectbox("Credit History", [1, 0])

ApplicantIncome = st.sidebar.number_input("Applicant Income", 0)
CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", 0)
LoanAmount = st.sidebar.number_input("Loan Amount", 0)
Loan_Amount_Term = st.sidebar.number_input("Loan Term", 0)

TotalIncome = ApplicantIncome + CoapplicantIncome

# ENCODING
# =========================
def encode(val, options):
    return options.index(val)

input_data = np.array([[
    encode(Gender, ["Male", "Female"]),
    encode(Married, ["Yes", "No"]),
    encode(Dependents, ["0", "1", "2", "3+"]),
    encode(Education, ["Graduate", "Not Graduate"]),
    encode(Self_Employed, ["Yes", "No"]),
    ApplicantIncome,
    CoapplicantIncome,
    LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    encode(Property_Area, ["Urban", "Semiurban", "Rural"]),
    TotalIncome  
]])

input_scaled = scaler.transform(input_data)

# =========================
# MAIN LAYOUT
# =========================
col1, col2 = st.columns([1,2])

# =========================
# PREDICTION
# =========================
with col1:
    st.subheader("🔍 Prediction")

    if st.button("Predict Loan Status"):
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.success(f"✅ Loan Approved (Confidence: {prob:.2f})")
        else:
            st.error(f"❌ Loan Not Approved (Confidence: {1-prob:.2f})")

# =========================
# FEATURE IMPORTANCE (EXPLAINABILITY)
# =========================
with col2:
    st.subheader("📊 Model Explanation")

    if hasattr(model, "feature_importances_"):
        features = ["Gender","Married","Education","Self_Employed","Property_Area",
                    "ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Term"]

        importances = model.feature_importances_

        fig, ax = plt.subplots()
        ax.barh(features, importances)
        ax.set_title("Feature Importance")
        st.pyplot(fig)

# =========================
# DATA INSIGHTS
# =========================
st.subheader("📈 Dataset Insights")

col3, col4 = st.columns(2)

with col3:
    fig1, ax1 = plt.subplots()
    df['Loan_Status'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_title("Loan Status Distribution")
    st.pyplot(fig1)

with col4:
    fig2, ax2 = plt.subplots()
    df['ApplicantIncome'].hist(ax=ax2)
    ax2.set_title("Applicant Income Distribution")
    st.pyplot(fig2)