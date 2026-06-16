# 🏦 Loan Prediction Dashboard

A Machine Learning-powered web application that predicts whether a loan application is likely to be approved based on applicant details, financial information, and credit history.

The application is built using **Python**, **Streamlit**, **Scikit-learn**, and **Matplotlib**, providing real-time predictions, model interpretability, EMI estimation, and dataset insights through an interactive dashboard.

---

## 🚀 Live Demo

🔗 **Application URL**

https://loan-prediction-app-corjpuxentyzfnb5khzkg3.streamlit.app/

---

## 📌 Project Overview

Financial institutions evaluate multiple factors before approving a loan application. This project uses a trained **Machine Learning model** to predict loan approval status based on applicant information such as:

- Gender
- Marital Status
- Number of Dependents
- Education Level
- Self Employment Status
- Applicant Income
- Co-applicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area

The dashboard provides instant predictions along with confidence scores and additional analytical insights.

---

## ✨ Features

### 🔍 Loan Approval Prediction
Predicts whether a loan application is:

- ✅ Approved
- ❌ Not Approved

with prediction confidence.

### 📊 Model Explanation
Displays feature importance/coefficient visualization to explain model behavior and decision-making.

### 🧮 EMI Calculator
Calculates:

- Monthly EMI
- Total Repayment Amount

based on the selected loan amount and tenure.

### 📈 Dataset Insights
Interactive charts showing:

- Loan Status Distribution
- Applicant Income Distribution

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Backend
- Python

### Machine Learning
- Scikit-learn
- Joblib

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib

---

## 📂 Project Structure

```text
Loan-prediction-app/
│
├── app.py
├── best_model.pkl
├── scaler.pkl
├── loan_prediction.csv
├── main.ipynb
├── requirements.txt
├── runtime.txt
├── LICENSE
└── README.md
```

---

## 🤖 Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Encoding
4. Feature Scaling
5. Model Training
6. Model Evaluation
7. Model Serialization using Joblib
8. Deployment using Streamlit Cloud

---

## 📋 Input Features

| Feature | Description |
|----------|------------|
| Gender | Male / Female |
| Married | Marital Status |
| Dependents | Number of Dependents |
| Education | Graduate / Not Graduate |
| Self Employed | Employment Status |
| Applicant Income | Monthly Income |
| Coapplicant Income | Co-applicant Income |
| Loan Amount | Requested Loan Amount |
| Loan Amount Term | Loan Repayment Period |
| Credit History | Credit History Record |
| Property Area | Urban / Semiurban / Rural |

---

## 📊 Dashboard Components

### Loan Prediction
Predicts loan approval status using a trained machine learning model.

### Model Explanation
Displays feature importance and coefficient-based analysis to improve transparency.

### EMI Estimator
Calculates estimated monthly installments and total repayment amount.

### Dataset Insights
Provides visual summaries of the training dataset.

---

## ▶️ Run Locally

### Clone Repository

```bash
git clone https://github.com/KAKUL23FE10CSE00261/Loan-prediction-app.git
cd Loan-prediction-app
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Application

```bash
streamlit run app.py
```

---

## 📈 Sample Output

- Loan Approval Prediction
- Prediction Confidence Score
- EMI Estimate
- Feature Importance Visualization
- Loan Dataset Insights

---

## 📷 Application Preview

### Dashboard Overview

- Applicant Details Input Panel
- Loan Approval Prediction
- EMI Calculator
- Model Explanation
- Dataset Insights

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Kakul Barsiya**

🎓 Manipal University Jaipur

🔗 GitHub: https://github.com/KAKUL23FE10CSE00261

🌐 Live App: https://loan-prediction-app-corjpuxentyzfnb5khzkg3.streamlit.app/

---

## ⭐ Future Improvements

- SHAP Explainable AI
- Multiple Model Comparison
- Loan Risk Scoring
- User Authentication
- Export Prediction Reports
- Enhanced Dashboard UI
- Cloud Database Integration

---

If you found this project useful, consider giving it a ⭐ on GitHub.
