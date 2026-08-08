import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pakistan Property Price Estimator", page_icon="🏠", layout="centered")

CITY_COORDS = {
    "Lahore": (31.467055, 74.321117),
    "Karachi": (24.927586, 67.121404),
    "Islamabad": (33.641052, 73.070155),
    "Rawalpindi": (33.561164, 73.075005),
    "Faisalabad": (31.425032, 73.106564),
}

PROPERTY_TYPES = ["House", "Flat", "Upper Portion", "Lower Portion", "Room", "Farm House", "Penthouse"]


@st.cache_resource
def load_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "property_price_model.joblib")
    return joblib.load(model_path)


pipeline = load_pipeline()

st.title("🏠 Pakistan Property Price Estimator")
st.write(
    "Estimate a property's sale price in PKR based on its city, type, size, and layout. "
    "Trained on 123,000+ real \"For Sale\" listings scraped from Zameen.com "
    "(XGBoost regression, R² ≈ 0.90 on held-out data)."
)

st.header("Property details")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("City", list(CITY_COORDS.keys()))
    property_type = st.selectbox("Property type", PROPERTY_TYPES)
    area_marla = st.number_input("Area (Marla)", min_value=1.0, max_value=2000.0, value=10.0, step=1.0)
    st.caption("1 Kanal = 20 Marla, for reference.")

with col2:
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3, step=1)
    baths = st.number_input("Bathrooms", min_value=0, max_value=20, value=3, step=1)

st.caption(
    f"Using average coordinates for {city} (fine-grained location within the city "
    "isn't collected by this simple form, so the citywide average is used)."
)

if st.button("Estimate Price", type="primary"):
    lat, lon = CITY_COORDS[city]
    bed_bath_ratio = bedrooms / (baths + 1)

    input_df = pd.DataFrame([{
        "city": city,
        "property_type": property_type,
        "area_marla": area_marla,
        "bedrooms": bedrooms,
        "baths": baths,
        "bed_bath_ratio": bed_bath_ratio,
        "latitude": lat,
        "longitude": lon,
    }])

    log_pred = pipeline.predict(input_df)[0]
    price_pred = np.expm1(log_pred)

    st.header("Estimated Price")
    st.success(f"💰 PKR {price_pred:,.0f}")
    st.caption(
        f"That's roughly PKR {price_pred/1e7:.2f} crore, or PKR {price_pred/area_marla:,.0f} per Marla."
    )

st.divider()
st.caption(
    "Model: XGBoost Regressor inside a scikit-learn Pipeline "
    "(StandardScaler + OneHotEncoder), trained on Zameen.com property listing data "
    "as the Neurofive Solutions ML Track capstone project. "
    "This is an estimate based on structured listing data only — it doesn't account for "
    "interior finishing, exact street, or current market conditions."
)
