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

ApplicantIncome = st.sidebar.number_input("Applicant Income (₹)", min_value=0, value=5000)
CoapplicantIncome = st.sidebar.number_input("Coapplicant Income (₹)", min_value=0, value=0)
LoanAmount = st.sidebar.number_input("Loan Amount (₹ thousands)", min_value=0, value=100)
Loan_Amount_Term = st.sidebar.number_input("Loan Term (months)", min_value=0, value=360)

TotalIncome = ApplicantIncome + CoapplicantIncome

# =========================
# INPUT VALIDATION
# =========================
warnings = []
if ApplicantIncome == 0:
    warnings.append("⚠️ Applicant Income is 0 — please enter a valid income.")
if LoanAmount == 0:
    warnings.append("⚠️ Loan Amount is 0 — please enter a valid loan amount.")
if Loan_Amount_Term == 0:
    warnings.append("⚠️ Loan Term is 0 — please enter a valid loan term.")

# =========================
# ENCODING
# =========================
def encode(val, options):
    return options.index(val)

FEATURE_NAMES = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area", "TotalIncome"
]

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
col1, col2 = st.columns([1, 2])

# =========================
# PREDICTION
# =========================
with col1:
    st.subheader("🔍 Prediction")

    if warnings:
        for w in warnings:
            st.warning(w)

    if st.button("Predict Loan Status"):
        if warnings:
            st.error("Please fix the warnings above before predicting.")
        else:
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]

            if pred == 1:
                st.success(f"✅ Loan Approved (Confidence: {prob:.2%})")
            else:
                st.error(f"❌ Loan Not Approved (Confidence: {(1 - prob):.2%})")

    # =========================
    # EMI CALCULATOR
    # =========================
    st.subheader("🧮 EMI Estimate")
    if LoanAmount > 0 and Loan_Amount_Term > 0:
        annual_rate = 0.085
        monthly_rate = annual_rate / 12
        loan_amt_rupees = LoanAmount * 1000
        emi = (loan_amt_rupees * monthly_rate * (1 + monthly_rate) ** Loan_Amount_Term) / \
              ((1 + monthly_rate) ** Loan_Amount_Term - 1)
        total_payment = emi * Loan_Amount_Term
        st.metric("Monthly EMI", f"₹ {emi:,.0f}")
        st.metric("Total Payment", f"₹ {total_payment:,.0f}")
        st.caption("Estimated at 8.5% annual interest rate")
    else:
        st.info("Enter Loan Amount and Term to see EMI estimate.")

# =========================
# MODEL EXPLANATION
# FIX: use coef_ for Logistic Regression, feature_importances_ for tree models
# =========================
with col2:
    st.subheader("📊 Model Explanation")

    if hasattr(model, "feature_importances_"):
        # Tree-based models (Random Forest, XGBoost, etc.)
        importances = model.feature_importances_
        colors = ["steelblue"] * len(FEATURE_NAMES)
        xlabel = "Importance Score"
        title = "Feature Importance"

    elif hasattr(model, "coef_"):
        # Linear models (Logistic Regression, SVM, etc.)
        importances = np.abs(model.coef_[0])
        colors = ["steelblue"] * len(FEATURE_NAMES)
        xlabel = "Coefficient Magnitude (Impact on Prediction)"
        title = "Feature Coefficients (Logistic Regression)"

    else:
        importances = None

    if importances is not None:
        sorted_idx = np.argsort(importances)
        sorted_names = [FEATURE_NAMES[i] for i in sorted_idx]
        sorted_vals = importances[sorted_idx]

        # Color positive/negative contributions
        if hasattr(model, "coef_"):
            raw_coefs = model.coef_[0]
            colors = ["green" if raw_coefs[i] > 0 else "red" for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.barh(sorted_names, sorted_vals, color=colors)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(xlabel)

        if hasattr(model, "coef_"):
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="green", label="Increases approval chance"),
                Patch(facecolor="red", label="Decreases approval chance")
            ]
            ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Model explanation is not available for this model type.")

# =========================
# DATA INSIGHTS
# =========================
st.subheader("📈 Dataset Insights")

col3, col4 = st.columns(2)

with col3:
    fig1, ax1 = plt.subplots()
    counts = df['Loan_Status'].value_counts()
    ax1.bar(["Approved (Y)", "Rejected (N)"], counts.values, color=["green", "red"])
    ax1.set_title("Loan Status Distribution")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

with col4:
    fig2, ax2 = plt.subplots()
    df['ApplicantIncome'].hist(ax=ax2, bins=30, color="steelblue", edgecolor="white")
    ax2.set_title("Applicant Income Distribution")
    ax2.set_xlabel("Income (₹)")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)
