from geopy.geocoders import Nominatim
import numpy as np

def conv_to_coordinates(address):
    geolocator = Nominatim(user_agent="uber_price_prediction_app")
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude)
    else:
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