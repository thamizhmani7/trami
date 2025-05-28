import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and label encoder
@st.cache_resource
def load_model():
    model = joblib.load("iris1.pkl")
    return model

model = load_model()

# Iris class names (optional: map directly instead of label encoder)
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Streamlit UI
st.title("🌸 Iris Species Predictor")
st.write("Enter flower measurements and predict its species!")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 0.0, 10.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 0.0, 10.0, 3.5)
petal_length = st.slider("Petal Length (cm)", 0.0, 10.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.0, 10.0, 0.2)

# Make prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
predicted_class = class_names[prediction]

# Display result
st.subheader("🔍 Prediction Result")
if predicted_class == 'Iris-setosa':
    st.success(f"Species: {predicted_class} — Good job! 🌼")
else:
    st.warning(f"Species: {predicted_class} — Bad job! 😅")

# Optional: Show raw prediction
# st.text(f"Raw model output: {prediction}")
