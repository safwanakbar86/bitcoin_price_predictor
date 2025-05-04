import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

rf_model = joblib.load('random_forest_model.pkl')
lr_model = joblib.load('linear_regression_model.pkl')
dl_model = load_model('neural_network_model.keras')
st.title("Bitcoin Closing Price Predictor")

model_choice = "Random Forest"
model_choice = st.selectbox("Select Model", ["Random Forest", "Linear Regression", "Deep Learning"])

st.subheader("Enter Feature Values")
open_val = st.number_input("Open Price", min_value=0.0)
high_val = st.number_input("High Price", min_value=0.0)
low_val = st.number_input("Low Price", min_value=0.0)
volume_val = st.number_input("Volume", min_value=0.0)
log_volume = np.log1p(volume_val)

input_features = np.array([[open_val, high_val, low_val, log_volume]])

if st.button("Predict Closing Price"):
    if model_choice == "Random Forest":
        prediction = rf_model.predict(input_features)[0]

    elif model_choice == "Linear Regression":
        try:
            scaler = joblib.load("scaler.pkl")
            input_scaled = scaler.transform(input_features)
            prediction = lr_model.predict(input_scaled)[0]
        except Exception as e:
            st.error(f"Scaler not found or error in prediction: {str(e)}")
            st.stop()

    elif model_choice == "Deep Learning":
        scaler = StandardScaler()
        try:
            scaler = joblib.load("scaler.pkl")
        except:
            st.error("Scaler not found. Please save and provide the same StandardScaler used in training.")
            st.stop()
            
        input_scaled = scaler.transform(input_features)
        prediction = dl_model.predict(input_scaled)[0][0]

    st.success(f"Predicted Closing Price: ${prediction:.2f}")