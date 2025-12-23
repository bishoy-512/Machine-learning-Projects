import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

# ========================
# Load trained model
# ========================

# مسار الموديل بالنسبة لـ gui.py
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

# تأكد من وجود الملف قبل التحميل
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)

# ========================
# Streamlit UI
# ========================

st.title("🚢 Titanic Survival Prediction")

st.header("Passenger Information")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
fare = st.number_input("Fare", min_value=0.0, value=0.0, step=0.1)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Collect input in DataFrame
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Fare": [fare],
    "Embarked": [embarked]
})

# Predict button
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][prediction]
        st.write(f"**Prediction:** {'Survived' if prediction==1 else 'Did not survive'}")
        st.write(f"**Confidence:** {probability*100:.2f}%")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
