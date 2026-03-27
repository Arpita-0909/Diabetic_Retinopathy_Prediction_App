import streamlit as st
import numpy as np
import pandas as pd
import joblib
import io
import matplotlib.pyplot as plt


# Configure page with background and custom font
st.set_page_config(page_title="Diabetic Retinopathy Predictor", page_icon="🧠", layout="centered")

# Adding custom CSS for modern design
st.markdown("""
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(to right, #00C6FF, #0072FF);
            color: #333;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            font-size: 18px;
            padding: 15px 30px;
            border: none;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }
        .stTextInput>div>div>input {
            border-radius: 12px;
            padding: 15px;
            border: 2px solid #d3d3d3;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .stTextInput>div>div>input:focus {
            border-color: #4CAF50;
        }
        .stSlider>div>div>div>input {
            border-radius: 12px;
            padding: 10px;
            font-size: 16px;
            border: 2px solid #d3d3d3;
            transition: all 0.3s ease;
        }
        .stSlider>div>div>div>input:focus {
            border-color: #4CAF50;
        }
        .result-card {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        .result-card:hover {
            transform: scale(1.02);
        }
        .result-card-high-risk {
            background-color: rgba(255, 99, 71, 0.85);
            color: white;
        }
        .result-card-low-risk {
            background-color: rgba(144, 238, 144, 0.85);
            color: white;
        }
        .header {
            font-size: 48px;
            color: white;
            font-weight: 700;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }
        .stMarkdown {
            font-size: 18px;
            text-align: center;
        }
        .stDownloadButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            padding: 14px 20px;
            transition: all 0.3s ease;
        }
        .stDownloadButton>button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Styling with Neumorphism Effect
st.sidebar.title("🔍 Model Selection")
st.sidebar.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 6px 10px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }
        .sidebar .sidebar-content .block-container {
            padding: 2rem 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Model selection and loading
model_files = {
    
    "Logistic Regression": "logreg_model.pkl"
  
}
selected_model_name = st.sidebar.radio("Choose a model:", list(model_files.keys()))
model = joblib.load(model_files[selected_model_name])

# Load scaler
scaler = joblib.load("scaler.pkl")

# Main Title
st.markdown("<h1 class='header'>🩺 Diabetic Retinopathy Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter patient details below and select a model to get a prediction.</p>", unsafe_allow_html=True)
st.markdown("---")

# Input fields with smooth animation and neumorphism
st.subheader("👤 Patient Information")
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", min_value=0, max_value=120, value=45, step=1, help="Select the patient's age")
    systolic_bp = st.slider("Systolic BP (mmHg)", min_value=50, max_value=250, value=120, step=1)
with col2:
    diastolic_bp = st.slider("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80, step=1)
    cholesterol = st.slider("Cholesterol (mg/dl)", min_value=50, max_value=400, value=180, step=1)

# On Predict
if st.button("🔮 Predict", help="Click to get the prediction based on your entered data"):
    input_data = np.array([[age, systolic_bp, diastolic_bp, cholesterol]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    result_text = "Retinopathy" if prediction == 1 else "No Retinopathy"

    if prediction == 1:
        st.markdown(f"""
            <div class="result-card result-card-high-risk">
                <b>⚠️ High Risk: {result_text}</b><br>
                Confidence: {probability*100:.2f}%<br>
                <i>Model Used: {selected_model_name}</i>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-card result-card-low-risk">
                <b>✅ Low Risk: {result_text}</b><br>
                Confidence: {probability*100:.2f}%<br>
                <i>Model Used: {selected_model_name}</i>
            </div>
        """, unsafe_allow_html=True)

    # Prepare result dict for CSV
    result_dict = {
        "Model Used": [selected_model_name],
        "Age": [age],
        "Systolic BP": [systolic_bp],
        "Diastolic BP": [diastolic_bp],
        "Cholesterol": [cholesterol],
        "Prediction": [result_text],
        "Confidence (%)": [f"{probability*100:.2f}"]
    }

    result_df = pd.DataFrame(result_dict)

    # CSV download
    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Result as CSV", csv_buffer.getvalue(), "retinopathy_prediction.csv", "text/csv")

    

    # Image generation
    def create_result_image(data):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis('off')
