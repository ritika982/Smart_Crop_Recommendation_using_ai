# app.py (Streamlit version)
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load models and scalers
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

st.title("🌾 Smart Crop Recommendation Using AI")

# User Inputs
N = st.number_input("Nitrogen", min_value=0.0, value=50.0)
P = st.number_input("Phosphorus", min_value=0.0, value=50.0)
K = st.number_input("Potassium", min_value=0.0, value=50.0)
temp = st.number_input("Temperature (°C)", min_value=0.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=100.0)

if st.button("🌱 Recommend Crop"):
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Apply scaling
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)

    # Prediction
    prediction = model.predict(final_features)

    crop_dict = {
        1: "rice", 2: "maize", 3: "chickpea", 4: "kidneybeans", 5: "pigeonpeas",
        6: "mothbeans", 7: "mungbeans", 8: "blackgram", 9: "lentil", 10: "pomegranate",
        11: "banana", 12: "mango", 13: "grapes", 14: "watermelon", 15: "muskmelon",
        16: "apple", 17: "orange", 18: "papaya", 19: "coconut", 20: "cotton",
        21: "jute", 22: "coffee"
    }

    if prediction[0] in crop_dict:
        result = f"✅ {crop_dict[prediction[0]]} is the best crop to be cultivated right there!"
    else:
        result = "❌ Sorry, could not determine the best crop for the given data."

    st.success(result)


