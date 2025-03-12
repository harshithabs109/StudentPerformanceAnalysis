import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Function to Predict Next Score
def predict_next_score():
    past_marks = np.array([60, 70, 75, 80, 85]).reshape(-1, 1)
    scores = np.array([65, 72, 78, 82, 87])
    future_test = np.array([[90]])

    model = LinearRegression()
    model.fit(past_marks, scores)
    predicted_score = model.predict(future_test)[0]
    return predicted_score

# Streamlit UI
st.title("📊 Student Performance Analysis")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Performance Prediction"])

# Prediction Page
if page == "Performance Prediction":
    st.write("### Predict Future Performance")
    
    predicted_score = predict_next_score()
    st.write(f"📈 **Predicted Next Score:** `{predicted_score:.2f}`")
    
    # Visualization
    past_tests = np.array([60, 70, 75, 80, 85, 90]).reshape(-1, 1)
    scores = np.array([65, 72, 78, 82, 87, predicted_score])

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(past_tests, scores, marker="o", linestyle="-", color="blue", label="Predicted Trend")
    ax.set_title("Student Performance Trend")
    ax.set_xlabel("Previous Test Scores")
    ax.set_ylabel("Actual & Predicted Marks")
    ax.legend()
    
    st.pyplot(fig)

