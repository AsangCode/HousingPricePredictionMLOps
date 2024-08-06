import streamlit as st
import pandas as pd
import os

from src.HousingPricePrediction.pipelines.prediction_pipeline import PredictPipeline, CustomData
from src.HousingPricePrediction.utils.utils import load_dataframe, load_object

# Load and prepare data
data = load_dataframe("notebooks/data", "housing.csv")
data.drop(['price'], axis=1, inplace=True)

st.write("""
# Housing Price Prediction App

This app predicts the **Housing Price** based on various features!
""")
st.write('---')

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features(data):
    # Ensure proper type handling for slider ranges and steps
    AREA = st.sidebar.slider('Area (in sq ft)', float(data.area.min()), float(data.area.max()), float(data.area.mean()))
    BEDROOMS = st.sidebar.slider('Number of Bedrooms', float(data.bedrooms.min()), float(data.bedrooms.max()), float(data.bedrooms.mean()))
    BATHROOMS = st.sidebar.slider('Number of Bathrooms', float(data.bathrooms.min()), float(data.bathrooms.max()), float(data.bathrooms.mean()))
    STORIES = st.sidebar.slider('Number of Stories', float(data.stories.min()), float(data.stories.max()), float(data.stories.mean()))
    
    MAINROAD = st.sidebar.selectbox('Located on Main Road', options=['yes', 'no'])
    GUESTROOM = st.sidebar.selectbox('Has Guestroom', options=['yes', 'no'])
    BASEMENT = st.sidebar.selectbox('Has Basement', options=['yes', 'no'])
    HOTWATERHEATING = st.sidebar.selectbox('Has Hot Water Heating', options=['yes', 'no'])
    AIRCONDITIONING = st.sidebar.selectbox('Has Air Conditioning', options=['yes', 'no'])
    PARKING = st.sidebar.slider('Parking Spaces', float(data.parking.min()), float(data.parking.max()), float(data.parking.mean()))
    PREFAREA = st.sidebar.selectbox('Preferred Area', options=['yes', 'no'])
    FURNISHINGSTATUS = st.sidebar.selectbox('Furnishing Status', options=['unfurnished', 'semi-furnished', 'furnished'])

    data = {
        'area': float(AREA),
        'bedrooms': float(BEDROOMS),
        'bathrooms': float(BATHROOMS),
        'stories': float(STORIES),
        'mainroad': str(MAINROAD),
        'guestroom': str(GUESTROOM),
        'basement': str(BASEMENT),
        'hotwaterheating': str(HOTWATERHEATING),
        'airconditioning': str(AIRCONDITIONING),
        'parking': float(PARKING),
        'prefarea': str(PREFAREA),
        'furnishingstatus': str(FURNISHINGSTATUS)
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features(data)


# Main Panel
# Print specified input parameters
st.header('Specified Input Parameters')
st.write(df)
st.write('---')

predict_pipeline = PredictPipeline()
prediction = predict_pipeline.predict(df)

st.header('Prediction of Housing Price')
st.write(f"The estimated price is: ${prediction[0]:,.2f}")
st.write('---')
