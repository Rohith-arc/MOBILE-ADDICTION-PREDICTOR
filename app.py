import streamlit as st
import joblib
import pandas as pd

# Load the saved model and scaler
model = joblib.load('smartphone_addiction_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Mapping for decoding the model's numeric predictions to labels
label_map = {0: "Low", 1: "Medium", 2: "High"}

# Function to process input and predict
def predict_addiction(input_data):
    # Ensure the input features are in the correct order and without 'Productivity_App_Usage'
    feature_columns = [
        "Screen_Time", "Social_Media_Usage", "Gaming_Time", "Sleep_Duration", 
        "Phone_Pickups", "Notifications_Received", "Battery_Consumption", "Age"
    ]
    
    # Convert input to a DataFrame with the same columns and order as training data
    input_df = pd.DataFrame([input_data], columns=feature_columns)

    # Scale the input using the same scaler used during training
    scaled_input = scaler.transform(input_df)

    # Predict using the model
    prediction_numeric = model.predict(scaled_input)
    prediction_label = label_map[prediction_numeric[0]]

    return prediction_numeric[0], prediction_label

# Streamlit app
st.title('Smartphone Addiction Prediction')

# Collect user inputs
age = st.number_input('Age', min_value=0, max_value=100, value=25)
screen_time = st.number_input('Screen Time (hours/day)', min_value=0, max_value=24, value=2)
social_media_usage = st.number_input('Social Media Usage (hours/day)', min_value=0, max_value=24, value=2)
gaming_time = st.number_input('Gaming Time (hours/day)', min_value=0, max_value=24, value=2)
sleep_duration = st.number_input('Sleep Duration (hours/day)', min_value=0, max_value=24, value=8)
phone_pickups = st.number_input('Phone Pickups (per day)', min_value=0, value=50)
notifications_received = st.number_input('Notifications Received (per day)', min_value=0, value=20)
battery_consumption = st.number_input('Battery Consumption (%)', min_value=0, max_value=100, value=50)

# Create the input data based on user inputs
input_data = [
    screen_time, social_media_usage, gaming_time, sleep_duration,
    phone_pickups, notifications_received, battery_consumption, age
]

# Predict when the user clicks the 'Predict' button
if st.button('Predict'):
    prediction_numeric, prediction_label = predict_addiction(input_data)
    st.write(f'Prediction: {prediction_label} (Numeric: {prediction_numeric})')
