import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('.ipynb_checkpoints/RFR_Uber_Price_Prediction_model.pkl')

st.title("Uber Price Prediction App")
st.write("Enter the ride details to predict the Uber fare.")

pickup_datetime = st.text_input("Pickup Date and Time (YYYY-MM-DD HH:MM:SS)", "2023-01-01 12:00:00")
trip_distance_km = st.number_input("Trip Distance (km)", min_value=0.0, step=0.1, value=5.0)
passenger_count = st.number_input("Number of Passengers", min_value=1, step=1, value=1)

if st.button("Predict Fare"):

    input_data = pd.DataFrame({
        'pickup_datetime': [pickup_datetime],
        'trip_distance_km': [trip_distance_km],
        'passenger_count': [passenger_count]
    })
    
    
    input_data['pickup_datetime'] = pd.to_datetime(input_data['pickup_datetime'])
    input_data['hour'] = input_data['pickup_datetime'].dt.hour
    input_data['day_of_week'] = input_data['pickup_datetime'].dt.dayofweek
    input_data['month'] = input_data['pickup_datetime'].dt.month
    

    features = input_data[['passenger_count', 'trip_distance_km', 'hour', 'day_of_week', 'month']]


    predicted_fare = model.predict(features)[0]
    
    st.success(f"Predicted Uber Fare: ${predicted_fare:.2f}")