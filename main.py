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
# Customer Information (Original Layout)
# ===========================
st.header("📋 Customer Information")

# ==== Row 1 ====
row1_col1, row1_col2, row1_col3 = st.columns([1,1,1])

with row1_col1:
    limit_bal = st.number_input("💰 Credit Limit Balance", min_value=0, max_value=1000000, step=1000, help="Range 0 to 1,000,000")

with row1_col2:
    age = st.number_input("🎂 Age", min_value=18, max_value=100, step=1, help="Range 18 to 100")

with row1_col3:
    credit_score = st.number_input("⭐ Credit Score", min_value=0, max_value=1000, step=1, help="Range 0 to 1000")

# ==== Row 2 ====
row2_col1, row2_col2, row2_col3 = st.columns([1,1,1])

with row2_col1:
    gender = st.selectbox(
        "🧑 Gender",
        ["👨 Male", "👩 Female"]
    )

with row2_col2:
    education = st.selectbox(
        "🎓 Education Level",
        ["🎓 Graduate School", "🏫 University", "🏢 High School", "📚 Others"]
    )

with row2_col3:
    marriage = st.selectbox(
        "💍 Marital Status",
        ["💑 Married", "🧑 Single", "🤝 Others"]
    )

# ===========================
# Payment History
# ===========================
st.divider()
st.header("📊 Payment History (Last 6 Months)")

st.markdown("""
**Repayment status explanation:**  
- `0` = Paid duly (on time)  
- `1` = Payment delay for 1 month  
- `2` = Payment delay for 2 months  
- … up to `8` months delay  
- `-1` or `-2` may indicate early payment or special cases
""")

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

# ===========================
# Bill & Payment Amounts
# ===========================
st.divider()
st.header("💵 Bill & Payment Amounts")

st.markdown("*Bill Amounts (Past 6 Months) — Range: 0 to 1,000,000 (NT$)*")
bill_cols = st.columns(3)
bill_amt_apr = bill_cols[0].number_input("Bill Apr", min_value=0, max_value=1000000, step=100)
bill_amt_may = bill_cols[1].number_input("Bill May", min_value=0, max_value=1000000, step=100)
bill_amt_jun = bill_cols[2].number_input("Bill Jun", min_value=0, max_value=1000000, step=100)

bill_cols2 = st.columns(3)
bill_amt_jul = bill_cols2[0].number_input("Bill Jul", min_value=0, max_value=1000000, step=100)
bill_amt_aug = bill_cols2[1].number_input("Bill Aug", min_value=0, max_value=1000000, step=100)
bill_amt_sep = bill_cols2[2].number_input("Bill Sep", min_value=0, max_value=1000000, step=100)

st.markdown("*Payment Amounts (Past 6 Months) — Range: 0 to 1,000,000 (NT$)*")
pay_cols = st.columns(3)
pay_amt_apr = pay_cols[0].number_input("Pay Apr", min_value=0, max_value=1000000, step=100)
pay_amt_may = pay_cols[1].number_input("Pay May", min_value=0, max_value=1000000, step=100)
pay_amt_jun = pay_cols[2].number_input("Pay Jun", min_value=0, max_value=1000000, step=100)

pay_cols2 = st.columns(3)
pay_amt_jul = pay_cols2[0].number_input("Pay Jul", min_value=0, max_value=1000000, step=100)
pay_amt_aug = pay_cols2[1].number_input("Pay Aug", min_value=0, max_value=1000000, step=100)
pay_amt_sep = pay_cols2[2].number_input("Pay Sep", min_value=0, max_value=1000000, step=100)

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

    if prob >= 0.60:   # High Risk
        st.markdown(
            f"""
            <div style="background-color:#ffcccc;padding:15px;border-radius:10px;">
                <h3 style="color:#b30000;">⚠ High Risk: This customer is <b>likely to DEFAULT</b></h3>
                <p style="color:#660000;font-size:16px;">
                <b>Risk Probability:</b> {prob:.2%}
                </p>
            </div>
            """, unsafe_allow_html=True
        )

    elif 0.40 <= prob < 0.60:   # Medium Risk
        st.markdown(
            f"""
            <div style="background-color:#fff3cd;padding:15px;border-radius:10px;">
                <h3 style="color:#856404;">⚠ Medium Risk: This customer has a <b>borderline chance of default</b></h3>
                <p style="color:#665c00;font-size:16px;">
                <b>Risk Probability:</b> {prob:.2%}
                </p>
            </div>
            """, unsafe_allow_html=True
        )

    else:   # Low Risk
        st.markdown(
            f"""
            <div style="background-color:#ccffcc;padding:15px;border-radius:10px;">
                <h3 style="color:#006600;">✅ Low Risk: This customer is <b>NOT likely to default</b></h3>
                <p style="color:#004d00;font-size:16px;">
                <b>Risk Probability:</b> {prob:.2%}
                </p>
            </div>
            """, unsafe_allow_html=True
        )
