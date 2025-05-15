import streamlit as st
import numpy as np
import pickle

# Load models and scaler
crop_model = pickle.load(open("crop_model.sav", "rb"))
crop_scaler = pickle.load(open("crop_scaler.sav", "rb"))

# Reverse mapping from encoded labels to crop names
label_to_crop = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut',
    6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon',
    11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil',
    16: 'blackgram', 17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas',
    20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
}

# Fertilizer suggestion logic
fertilizer_dict = {
    "N": "Use Urea or Ammonium Sulphate (Nitrogen-rich)",
    "P": "Use Single Super Phosphate or DAP (Phosphorus-rich)",
    "K": "Use Muriate of Potash (Potassium-rich)",
    "Balanced": "Your soil is balanced. Use compost or organic fertilizers."
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
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    scaled_input = crop_scaler.transform(input_data)
    prediction = crop_model.predict(scaled_input)

    # Convert numeric label to crop name
    label_index = int(prediction[0])
    predicted_crop = label_to_crop.get(label_index, "Unknown").capitalize()

    # Fertilizer suggestion based on deficiency logic
    npk = {"N": N, "P": P, "K": K}
    max_deficiency = max(npk, key=lambda x: 90 - npk[x] if 90 - npk[x] > 15 else 0)
    fertilizer = fertilizer_dict.get(max_deficiency, fertilizer_dict["Balanced"])

    st.success(f"🌾 Recommended Crop: **{predicted_crop}**")
    st.info(f"🧪 Suggested Fertilizer Strategy: **{fertilizer}**")
