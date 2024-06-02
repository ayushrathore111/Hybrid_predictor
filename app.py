import streamlit as st
import pandas as pd
import joblib

# Load the trained models
rf_model = joblib.load('/home/proayush/Desktop/ds/random_forest_model.joblib')
gpr_model = joblib.load('/home/proayush/Desktop/ds/gpr_model.joblib')

# Title of the web app
st.title('Streamlit App for Stream Prediction')

# Sidebar with input fields
st.sidebar.title('Input Features')
U = st.sidebar.slider('Velocity (m/s)', min_value=0.09, max_value=0.28, step=0.01, value=0.15)
H = st.sidebar.slider('Water Depth (m)', min_value=0.02, max_value=0.28, step=0.01, value=0.13)
D = st.sidebar.slider('Streambed Diameter (m)', min_value=0.06, max_value=0.22, step=0.01, value=0.11)
Fr = st.sidebar.slider('Froude Number', min_value=0.07, max_value=1.33, step=0.01, value=0.24)
d50 = st.sidebar.slider('Median Grain Size (m)', min_value=0.0, max_value=0.1, step=0.01, value=0.0)
HD_ratio = st.sidebar.slider('Water Depth to Streambed Diameter Ratio', min_value=0.4, max_value=6.8, step=0.01, value=1.182)
Dd50_ratio = st.sidebar.slider('Streambed Diameter to Median Grain Size Ratio', min_value=0.01, max_value=2.29, step=0.01, value=0.05)
DsD_ratio = st.sidebar.slider('Froude Number to Streambed Diameter Ratio', min_value=0.01, max_value=2.29, step=0.01, value=0.18)

# Function to make predictions using the models
def make_predictions(rf_model, gpr_model, U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio):
    # Make predictions using the models
    prediction_rf = rf_model.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio]])[0]
    prediction_gpr = gpr_model.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio]])[0]
    return prediction_rf, prediction_gpr

# Make predictions using the input values
prediction_rf, prediction_gpr = make_predictions(rf_model, gpr_model, U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio)

# Display predictions
st.write('### Random Forest Prediction (ds):', prediction_rf)
st.write('### Gaussian Process Prediction (ds):', prediction_gpr)

