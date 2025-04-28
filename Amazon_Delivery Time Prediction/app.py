import streamlit as st
import pandas as pd
import numpy as np
import joblib
from geopy.distance import geodesic

# Load your Random Forest model
model = joblib.load('best_model.pkl')

# Page config
st.set_page_config(page_title="Amazon Delivery Time Prediction", page_icon="🚚")

# Title
st.title("🚚 Amazon Delivery Time Prediction")

# Sidebar
st.sidebar.header("Enter Order Details")

store_lat = st.sidebar.number_input("Store Latitude", format="%.6f")
store_long = st.sidebar.number_input("Store Longitude", format="%.6f")
drop_lat = st.sidebar.number_input("Drop Latitude", format="%.6f")
drop_long = st.sidebar.number_input("Drop Longitude", format="%.6f")

agent_age = st.sidebar.slider("Agent Age", 18, 70)
agent_rating = st.sidebar.slider("Agent Rating", 1.0, 5.0, 0.1)

weather = st.sidebar.selectbox("Weather Condition", ['Sunny', 'Cloudy', 'Rainy', 'Stormy', 'Foggy'])
traffic = st.sidebar.selectbox("Traffic Condition", ['Low', 'Medium', 'High', 'Jam'])
vehicle = st.sidebar.selectbox("Vehicle Type", ['Bike', 'Car', 'Van'])
area = st.sidebar.selectbox("Delivery Area", ['Urban', 'Metropolitan'])
category = st.sidebar.selectbox("Product Category", ['Electronics', 'Clothing', 'Grocery', 'Home', 'Others'])

# Distance Calculation
def calculate_distance(lat1, lon1, lat2, lon2):
    loc1 = (lat1, lon1)
    loc2 = (lat2, lon2)
    return geodesic(loc1, loc2).km

distance_km = calculate_distance(store_lat, store_long, drop_lat, drop_long)

# Prepare input
def create_input():
    input_data = {
        'Agent_Age': agent_age,
        'Agent_Rating': agent_rating,
        'Distance_km': distance_km,
        'Order_DayOfWeek': pd.Timestamp.now().dayofweek,
        'Order_Month': pd.Timestamp.now().month,
    }

    # One-hot encoding
    features = {
        'Sunny', 'Cloudy', 'Rainy', 'Stormy', 'Foggy',
        'Low', 'Medium', 'High', 'Jam',
        'Bike', 'Car', 'Van',
        'Urban', 'Metropolitan',
        'Electronics', 'Clothing', 'Grocery', 'Home', 'Others'
    }

    for f in features:
        input_data[f"Weather_{f}"] = 1 if weather == f else 0
        input_data[f"Traffic_{f}"] = 1 if traffic == f else 0
        input_data[f"Vehicle_{f}"] = 1 if vehicle == f else 0
        input_data[f"Area_{f}"] = 1 if area == f else 0
        input_data[f"Category_{f}"] = 1 if category == f else 0

    return pd.DataFrame([input_data])

# Prediction
if st.button("Predict Delivery Time 🚀"):
    input_df = create_input()  # Generate the input dataframe

    # Dynamically get the expected feature names from the trained model
    expected_features = model.feature_names_in_

    # Ensure the input data has all the expected features, filling missing ones with 0
    input_df = input_df.reindex(columns=expected_features, fill_value=0)
    
    with st.spinner('Predicting...'):
        prediction = model.predict(input_df)[0]
    st.success(f"Estimated Delivery Time: {prediction:.2f} hours ⏳")

st.caption("Built with ❤️ using Streamlit and Machine Learning")
