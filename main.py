import streamlit as st
import pickle
import numpy as np

# ===========================
# Load trained pipeline
# ===========================
with open("Model/creditcard_defaulters_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# ===========================
# App Layout
# ===========================
st.set_page_config(page_title="Credit Card Default Predictor", page_icon="💳", layout="centered")

st.title("💳 Credit Card Default Prediction")
st.markdown("""
Welcome to the *Credit Risk Analyzer*!  
Fill in the details below to check whether a customer is *likely to default* on their next credit card payment.  
""")

st.divider()

# ===========================
# Inputs Section
# ===========================

st.header("📋 Customer Information")

col1, col2 = st.columns(2)

with col1:
    limit_bal = st.number_input("💰 Credit Limit Balance", min_value=0, max_value=1000000, step=1000, help="Total credit limit assigned to the customer.")
    age = st.number_input("🎂 Age", min_value=18, max_value=100, step=1, help="Age of the customer.")
    gender = st.radio("🧑 Gender", ["Male", "Female"])
    education = st.radio("🎓 Education Level", ["Graduate School", "University", "High School", "Others"])
    marriage = st.radio("💍 Marital Status", ["Married", "Single", "Others"])

with col2:
    credit_score = st.number_input("⭐ Credit Score", min_value=0, max_value=1000, step=1, help="Customer credit score (engineered feature).")

st.divider()
st.header("📊 Payment History (Last 6 Months)")

col3, col4, col5 = st.columns(3)

with col3:
    pay_sep = st.number_input("Repayment Sep", min_value=-2, max_value=8, step=1)
    pay_aug = st.number_input("Repayment Aug", min_value=-2, max_value=8, step=1)

with col4:
    pay_jul = st.number_input("Repayment Jul", min_value=-2, max_value=8, step=1)
    pay_jun = st.number_input("Repayment Jun", min_value=-2, max_value=8, step=1)

with col5:
    pay_may = st.number_input("Repayment May", min_value=-2, max_value=8, step=1)
    pay_apr = st.number_input("Repayment Apr", min_value=-2, max_value=8, step=1)

st.divider()
st.header("💵 Bill & Payment Amounts")

st.markdown("*Bill Amounts (Past 6 Months)*")
bill_cols = st.columns(3)
bill_amt_apr = bill_cols[0].number_input("Bill Apr", min_value=0, step=100)
bill_amt_may = bill_cols[1].number_input("Bill May", min_value=0, step=100)
bill_amt_jun = bill_cols[2].number_input("Bill Jun", min_value=0, step=100)

bill_cols2 = st.columns(3)
bill_amt_jul = bill_cols2[0].number_input("Bill Jul", min_value=0, step=100)
bill_amt_aug = bill_cols2[1].number_input("Bill Aug", min_value=0, step=100)
bill_amt_sep = bill_cols2[2].number_input("Bill Sep", min_value=0, step=100)

st.markdown("*Payment Amounts (Past 6 Months)*")
pay_cols = st.columns(3)
pay_amt_apr = pay_cols[0].number_input("Pay Apr", min_value=0, step=100)
pay_amt_may = pay_cols[1].number_input("Pay May", min_value=0, step=100)
pay_amt_jun = pay_cols[2].number_input("Pay Jun", min_value=0, step=100)

pay_cols2 = st.columns(3)
pay_amt_jul = pay_cols2[0].number_input("Pay Jul", min_value=0, step=100)
pay_amt_aug = pay_cols2[1].number_input("Pay Aug", min_value=0, step=100)
pay_amt_sep = pay_cols2[2].number_input("Pay Sep", min_value=0, step=100)

# ===========================
# Encode categorical variables
# ===========================
Gender_Female = 1 if gender == "Female" else 0
Education_University = 1 if education == "University" else 0
Education_HighSchool = 1 if education == "High School" else 0
Education_Others = 1 if education == "Others" else 0
Marriage_Single = 1 if marriage == "Single" else 0
Marriage_Others = 1 if marriage == "Others" else 0

# ===========================
# Arrange input in pipeline order
# ===========================
input_data = np.array([[
    limit_bal,
    age,
    pay_sep, pay_aug, pay_jul, pay_jun, pay_may, pay_apr,
    bill_amt_apr, bill_amt_may, bill_amt_jun, bill_amt_jul, bill_amt_aug, bill_amt_sep,
    pay_amt_apr, pay_amt_may, pay_amt_jun, pay_amt_jul, pay_amt_aug, pay_amt_sep,
    credit_score,
    Gender_Female,
    Education_University, Education_HighSchool, Education_Others,
    Marriage_Single, Marriage_Others
]])

# ===========================
# Prediction Section
# ===========================
st.divider()
st.header("🔮 Prediction Result")

if st.button("Run Prediction 🚀"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠ High Risk: This customer is *likely to DEFAULT.\n\nRisk Probability:* {prob:.2%}")
    else:
        st.success(f"✅ Low Risk: This customer is *NOT likely to default.\n\nRisk Probability:* {prob:.2%}")