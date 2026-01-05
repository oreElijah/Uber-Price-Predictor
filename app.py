import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import joblib

model = joblib.load('.ipynb_checkpoints/RFR_Uber_Price_Prediction_model.pkl')

st.title("Uber Price Prediction App")
st.write("Enter the ride details to predict the Uber fare.")

pickup_datetime = st.text_input("Pickup Date and Time (YYYY-MM-DD HH:MM:SS)", "2023-01-01 12:00:00")
pickup_address = st.text_input("Pickup Address", "1600 Amphitheatre Parkway, Mountain View, CA")
dropoff_address = st.text_input("Dropoff Address", "1 Infinite Loop, Cupertino, CA")
passenger_count = st.number_input("Number of Passengers", min_value=1, step=1, value=1)

def conv_to_coordinates(address):
    geolocator = Nominatim(user_agent="uber_price_prediction_app")
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude)
    else:
        st.error(f"Could not geocode address: {address}")
        return (None, None)
    
def Haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lat2, lon1, lon2 = map(
        np.radians, [lat1, lat2, lon1, lon2]
    ) 
    dlat = lat2-lat1
    dlon = lon2-lon1

    a = (np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2)

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    d = R * c
    return d

if st.button("Predict Fare"):
    pickup_coords = conv_to_coordinates(pickup_address)
    dropoff_coords = conv_to_coordinates(dropoff_address)

    if pickup_coords[0] is not None and dropoff_coords[0] is not None:
        trip_distance_km = Haversine(pickup_coords[0], pickup_coords[1], dropoff_coords[0], dropoff_coords[1])
    else:
        trip_distance_km = st.number_input("Trip Distance (km)", min_value=0.0, step=0.1, value=5.0)
        st.warning("Using manual trip distance input.")
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