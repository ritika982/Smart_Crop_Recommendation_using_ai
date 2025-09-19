import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

st.title("🌾 Smart Crop Recommendation System")

# Input fields
N = st.number_input("Nitrogen", min_value=0.0, value=50.0)
P = st.number_input("Phosphorus", min_value=0.0, value=30.0)
K = st.number_input("Potassium", min_value=0.0, value=20.0)
temp = st.number_input("Temperature (°C)", min_value=0.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, value=70.0)
ph = st.number_input("pH", min_value=0.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=200.0)

if st.button("Predict Crop"):
    # Prepare features
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Scale features
    scaled_features = ms.transform(single_pred)

    # Predict
    prediction = model.predict(scaled_features)

    crop_dict = {
        1: "rice", 2: "maize", 3: "chickpea", 4: "kidneybeans",
        5: "pigeonpeas", 6: "mothbeans", 7: "mungbeans", 8: "blackgram",
        9: "lentil", 10: "pomegranate", 11: "banana", 12: "mango",
        13: "grapes", 14: "watermelon", 15: "muskmelon", 16: "apple",
        17: "orange", 18: "papaya", 19: "coconut", 20: "cotton",
        21: "jute", 22: "coffee"
    }

    if prediction[0] in crop_dict:
        st.success(f"✅ Recommended Crop: **{crop_dict[prediction[0]]}**")
    else:
        st.error("❌ Sorry, could not determine the best crop with the given data.")




