import numpy as np
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

@st.cache_data(ttl=86400)
def conv_to_coordinates(address):
    geolocator = Nominatim(user_agent="uber_price_prediction_app")
    try:
        location = geolocator.geocode(address, timeout=10)
        if location:
            return (location.latitude, location.longitude)
        return (None, None)
    except (GeocoderTimedOut, GeocoderUnavailable):
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
