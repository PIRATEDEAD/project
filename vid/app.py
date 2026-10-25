#Gender-->1 Female 0 male
#Churmn-->1 yes 0 no
#scaler is exported as scaler.pkl
#model. is exported as model.pkl
#order of the x -->Age', 'Gender', 'Tenure', 'MonthlyCharges'],

import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Customer Churn Prediction App")
st.divider()
st.write("Please enter the values and hit the predict button for prediction.")
st.divider()

age = st.number_input("Enter the Age", min_value=18, max_value=100, value=30)

tenure = st.number_input("Enter the Tenure", min_value=0, max_value=130, value=10)

monthly_charges = st.number_input("Enter the Monthly Charges", min_value=30, max_value=150)

gender = st.selectbox("Select the Gender", options=["Male", "Female"])

st.divider()

predictbutton = st.button("Predict")
if predictbutton:
    gender = 1 if gender == "Female" else 0.       # Convert Gender to the same encoding used during training.


# Keep feature order consistent with the training notebook:
# Age, Gender, Tenure, MonthlyCharges.
    X = [age, gender, tenure, monthly_charges]  


# Apply the training scaler before making a prediction.
    X1 = np.array(X)

    X_array=  scaler.transform([X1])

# Convert the model's numeric output into a readable label.
    prediction = model.predict(X_array)[0]

    predicted = "Churn" if prediction == 1 else "Not Churn"

    st.write(f"Predicted: {predicted}")

else:
    st.write("Please enter the values and hit the predict button for prediction.")

