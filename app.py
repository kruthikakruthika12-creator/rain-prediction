
import streamlit as st
import pandas as pd
import joblib

# Page settings
st.set_page_config(
    page_title="Rain Prediction",
    page_icon="🌧️",
    layout="centered"
)

# Load model
model = joblib.load("rain_model.pkl")

# Title
st.title("🌧️ Rain Prediction System")

st.write(
    "Enter the current weather conditions to predict whether it will rain."
)

st.divider()

# Input section
st.header("🌡️ Enter Weather Conditions")

humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

pressure = st.number_input(
    "Pressure (hPa)",
    min_value=900.0,
    max_value=1100.0,
    value=1005.0
)

wind_speed = st.number_input(
    "Wind Speed",
    min_value=0.0,
    max_value=150.0,
    value=12.0
)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=-20.0,
    max_value=60.0,
    value=24.0
)

# Prediction button
if st.button("🔮 Predict Rain", use_container_width=True):

    # Prepare input
    input_data = pd.DataFrame({
        "Humidity": [humidity],
        "Pressure": [pressure],
        "WindSpeed": [wind_speed],
        "Temperature": [temperature]
    })

    # Predict
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    st.header("📊 Prediction Result")

    if prediction == 1:

        st.success("🌧️ RAIN PREDICTED")

        st.write(
            f"Rain Probability: {probability * 100:.2f}%"
        )

    else:

        st.info("☀️ NO RAIN PREDICTED")

        st.write(
            f"Rain Probability: {probability * 100:.2f}%"
        )

    # Display entered conditions
    st.subheader("Weather Conditions Entered")

    result = pd.DataFrame({
        "Condition": [
            "Humidity",
            "Pressure",
            "Wind Speed",
            "Temperature"
        ],
        "Value": [
            f"{humidity:.1f} %",
            f"{pressure:.1f} hPa",
            f"{wind_speed:.1f}",
            f"{temperature:.1f} °C"
        ]
    })

    st.table(result)
