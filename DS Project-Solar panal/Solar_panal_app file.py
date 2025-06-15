import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the saved model
def load_model():
    with open("solar_power_model.pkl", "rb") as f:
        model, scaler = pickle.load(f)
    return model, scaler

# Predict function
def predict_power(input_data):
    model, scaler = load_model()
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)
    return prediction[0]

# Streamlit deployment
import streamlit as st

def main():
    st.title("Solar Power Generation Prediction")
    
    # Input fields
    distance_to_solar_noon = st.number_input("Distance to Solar Noon")
    temperature = st.number_input("Temperature")
    wind_direction = st.number_input("Wind Direction")
    sky_cover = st.number_input("Sky Cover")
    visibility = st.number_input("Visibility")
    humidity = st.number_input("Humidity")
    avg_wind_speed = st.number_input("Average Wind Speed (Period)")
    avg_pressure = st.number_input("Average Pressure (Period)")
    
    if st.button("Predict Power Generation"):
        input_data = [
            distance_to_solar_noon, temperature, wind_direction, 
            sky_cover, visibility, humidity, avg_wind_speed, avg_pressure
        ]
        prediction = predict_power(input_data)
        st.success(f"Predicted Power Generation: {prediction:.2f} W")

if __name__ == "__main__":
    main()