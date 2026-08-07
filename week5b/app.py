import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")


import os

@st.cache_resource
def load_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "churn_pipeline.joblib")
    return joblib.load(model_path)


pipeline = load_pipeline()

SERVICE_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]

st.title("📉 Customer Churn Predictor")
st.write(
    "Enter a customer's details below to predict whether they're likely to churn. "
    "This app uses a scikit-learn pipeline (StandardScaler + OneHotEncoder + "
    "Logistic Regression) trained on the Telco Customer Churn dataset."
)

st.header("Customer details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior citizen?", ["No", "Yes"])
    partner = st.selectbox("Has a partner?", ["No", "Yes"])
    dependents = st.selectbox("Has dependents?", ["No", "Yes"])
    tenure = st.slider("Tenure (months as a customer)", 0, 72, 12)
    phone_service = st.selectbox("Phone service?", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple lines?", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])

with col2:
    paperless_billing = st.selectbox("Paperless billing?", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
    total_charges = st.number_input("Total charges to date ($)", min_value=0.0, max_value=10000.0, value=840.0, step=10.0)
    online_security = st.selectbox("Online security?", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online backup?", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device protection?", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech support?", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV?", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming movies?", ["No", "Yes", "No internet service"])

if st.button("Predict", type="primary"):
    raw_input = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # Engineered features - must match training exactly
    raw_input["NumServices"] = sum(
        1 for col in SERVICE_COLS if raw_input[col] == "Yes"
    )
    raw_input["AvgMonthlySpend"] = raw_input["TotalCharges"] / (raw_input["tenure"] + 1)

    input_df = pd.DataFrame([raw_input])

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    st.header("Result")
    if prediction == 1:
        st.error(f"⚠️ Likely to churn — {probability*100:.1f}% predicted probability")
    else:
        st.success(f"✅ Not likely to churn — {probability*100:.1f}% predicted probability of churn")

    st.progress(min(int(probability * 100), 100))

st.caption(
    "Model: Logistic Regression inside a scikit-learn Pipeline "
    "(StandardScaler + OneHotEncoder). Trained on the Telco Customer Churn dataset "
    "as part of the Neurofive Solutions ML Track."
)
