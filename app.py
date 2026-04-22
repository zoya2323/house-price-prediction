import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load model
model = joblib.load("models/model.pkl")

# Sidebar
st.sidebar.title("🏠 House Price Predictor")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app predicts house prices based on key house features using a trained Machine Learning model."
)

st.sidebar.markdown("### Model Used")
st.sidebar.success("Random Forest Regressor")

# Main Title
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🏡 House Price Prediction App</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Enter the details below to estimate your house price.</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# Input Section
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("⭐ Overall Quality", 1, 10, 5)
    garage_cars = st.number_input("🚗 Garage Capacity", value=1)

with col2:
    gr_liv_area = st.number_input("📏 Living Area (sq ft)", value=1500)
    total_bsmt_sf = st.number_input("🏠 Basement Area", value=800)

st.markdown("")

# Prediction button
if st.button("🔍 Predict Price"):

    input_data = pd.DataFrame({
        "OverallQual": [overall_qual],
        "GrLivArea": [gr_liv_area],
        "GarageCars": [garage_cars],
        "TotalBsmtSF": [total_bsmt_sf]
    })

    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_data)[0]

    st.markdown("## 📊 Prediction Result")
    
    st.metric(label="🏡 Estimated House Price", value=f"₹ {round(prediction, 2)}")

st.markdown("---")

