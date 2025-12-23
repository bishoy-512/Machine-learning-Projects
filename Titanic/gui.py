import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Titanic Survival Prediction")

st.title("🚢 Titanic Survival Prediction")

# Load model
import os
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.subheader("Passenger Information")

Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Fare = st.number_input("Fare", min_value=0.0, step=1.0)
Embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# Create DataFrame (VERY IMPORTANT: نفس الأعمدة بالضبط)
input_data = pd.DataFrame({
    "Pclass": [Pclass],
    "Sex": [Sex],
    "Fare": [Fare],
    "Embarked": [Embarked]
})

if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Passenger is likely to SURVIVE ✅ (Probability: {probability:.2f})")
    else:
        st.error(f"Passenger is NOT likely to survive ❌ (Probability: {probability:.2f})")

