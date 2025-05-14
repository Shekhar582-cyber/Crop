import streamlit as st
import numpy as np
import pickle

# Load models and scaler
crop_model = pickle.load(open("crop_model.sav", "rb"))
crop_scaler = pickle.load(open("crop_scaler.sav", "rb"))

# Fertilizer dictionary (simple mock logic)
fertilizer_dict = {
    "N": "Use Urea or Ammonium Sulphate (Nitrogen-rich)",
    "P": "Use Single Super Phosphate or DAP (Phosphorus-rich)",
    "K": "Use Muriate of Potash (Potassium-rich)",
    "Balanced": "Your soil is balanced. Use compost or organic fertilizers."
}

# Label-to-crop mapping (used if model returns integers)
label_to_crop = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut',
    5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans',
    10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans', 14: 'mungbean',
    15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas',
    19: 'pomegranate', 20: 'rice', 21: 'watermelon'
}

st.set_page_config(page_title="🌾 Crop & Fertilizer Recommender", layout="centered")
st.title("🌿 AI-Based Crop and Fertilizer Recommendation System")

st.markdown("Enter the following parameters:")

# Input fields
N = st.number_input("Nitrogen (N)", 0, 150, step=1)
P = st.number_input("Phosphorus (P)", 0, 150, step=1)
K = st.number_input("Potassium (K)", 0, 150, step=1)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0)
ph = st.number_input("pH", 0.0, 14.0)
rainfall = st.number_input("Rainfall (mm)", 0.0, 400.0)

if st.button("🌱 Predict Crop and Fertilizer"):
    # Crop Prediction
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    scaled_input = crop_scaler.transform(input_data)
    crop_prediction = crop_model.predict(scaled_input)
    prediction_value = crop_prediction[0]

    # Detect type and map if numeric
    if isinstance(prediction_value, (int, np.integer)):
        predicted_crop = label_to_crop.get(int(prediction_value), "Unknown").capitalize()
    else:
        predicted_crop = str(prediction_value).capitalize()

    # Fertilizer Suggestion
    npk = {"N": N, "P": P, "K": K}
    max_deficiency = max(npk, key=lambda x: 90 - npk[x] if 90 - npk[x] > 15 else 0)
    fertilizer = fertilizer_dict.get(max_deficiency, fertilizer_dict["Balanced"])

    # Display Output
    st.success(f"🌾 Recommended Crop: **{predicted_crop}**")
    st.info(f"🧪 Suggested Fertilizer Strategy: **{fertilizer}**")
