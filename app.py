import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Road Accident Severity Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- CSS FOR STYLING ---
# We use modern aesthetics: subtle gradients, nice cards, and clean typography.
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Title Style */
    h1 {
        font-family: 'Inter', sans-serif;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0px !important;
    }
    
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1em;
        margin-bottom: 40px;
    }
    
    /* Prediction Box Styling */
    .prediction-card {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 30px;
        transition: transform 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
    }
    
    .severity-minor {
        color: #27ae60;
        font-weight: bold;
        font-size: 2em;
    }
    .severity-moderate {
        color: #f39c12;
        font-weight: bold;
        font-size: 2em;
    }
    .severity-severe {
        color: #d35400;
        font-weight: bold;
        font-size: 2em;
    }
    .severity-fatal {
        color: #c0392b;
        font-weight: bold;
        font-size: 2em;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
</style>
""", unsafe_allow_html=True)

def load_models_and_transformers():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        rf_model = joblib.load(os.path.join(current_dir, 'rf_model.pkl'))
        svm_model = joblib.load(os.path.join(current_dir, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))
        label_encoders = joblib.load(os.path.join(current_dir, 'label_encoders.pkl'))
        target_encoder = joblib.load(os.path.join(current_dir, 'target_encoder.pkl'))
        return rf_model, svm_model, scaler, label_encoders, target_encoder
    except Exception as e:
        return None, None, None, None, None

def main():
    st.markdown("<h1>Road Accident Severity Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Using Data Mining & Machine Learning to predict outcomes</div>", unsafe_allow_html=True)

    rf_model, svm_model, scaler, label_encoders, target_encoder = load_models_and_transformers()

    if rf_model is None or svm_model is None:
        st.error("⚠️ Models not found. Please run the training script (`train_model.py`) first to generate the models.")
        return

    # We will use both models for prediction to provide a consensus

    # --- INPUT FORM ---
    with st.container():
        st.write("### 📋 Enter Accident Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            weather = st.selectbox("Weather Condition", ['Clear', 'Rain', 'Snow', 'Fog'])
            road_type = st.selectbox("Road Type", ['Highway', 'City Street', 'Rural Road'])
            time_of_day = st.selectbox("Time of Day", ['Morning', 'Afternoon', 'Evening', 'Night'])
            
        with col2:
            vehicle_type = st.selectbox("Vehicle Type", ['Car', 'Truck', 'Motorcycle', 'Bus'])
            speed_limit = st.slider("Speed Limit (mph)", min_value=15, max_value=85, value=45, step=5)
            driver_age = st.number_input("Driver Age", min_value=16, max_value=100, value=35)

    # --- PREDICTION LOGIC ---
    if st.button("Predict Severity 🎯", use_container_width=True):
        with st.spinner('Analyzing patterns...'):
            # Prepare input data
            input_data = pd.DataFrame({
                'Weather_Condition': [weather],
                'Road_Type': [road_type],
                'Speed_Limit': [speed_limit],
                'Time_of_Day': [time_of_day],
                'Vehicle_Type': [vehicle_type],
                'Driver_Age': [driver_age]
            })

            # Encode categorical features
            for col in ['Weather_Condition', 'Road_Type', 'Time_of_Day', 'Vehicle_Type']:
                le = label_encoders[col]
                # Handle unseen labels by mapping them to an existing one (fallback)
                if input_data[col][0] not in le.classes_:
                    input_data[col] = le.transform([le.classes_[0]])
                else:
                    input_data[col] = le.transform(input_data[col])

            # Scale features
            input_scaled = scaler.transform(input_data)

            # Predict with both models
            rf_pred_encoded = rf_model.predict(input_scaled)
            svm_pred_encoded = svm_model.predict(input_scaled)
            
            rf_label = target_encoder.inverse_transform(rf_pred_encoded)[0]
            svm_label = target_encoder.inverse_transform(svm_pred_encoded)[0]
            
            # Determine overall condition
            high_risk_labels = ['Severe', 'Fatal']
            is_high_risk = (rf_label in high_risk_labels) or (svm_label in high_risk_labels)
            
            # Display explicitly if it is safe or a risk alert
            if is_high_risk:
                st.error("🚨 **RISK ALERT: IT IS NOT SAFE TO DRIVE!**")
            else:
                st.success("✅ **DRIVE SAFE: CONDITIONS ARE FAVORABLE!**")
                
            st.markdown(f"""
            <div class='prediction-card'>
                <h3 style='color: #34495e;'>Algorithm Predictions:</h3>
                <p><b>Random Forest:</b> <span class='severity-{rf_label.lower()}'>{rf_label}</span></p>
                <p><b>Support Vector Machine:</b> <span class='severity-{svm_label.lower()}'>{svm_label}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display dynamic recommendations based on condition
            recommendations = []
            if weather in ['Snow', 'Fog', 'Rain']:
                recommendations.append(f"Visibility and traction are reduced due to {weather}.")
            if speed_limit >= 65:
                recommendations.append(f"High speeds ({speed_limit} mph) significantly increase accident severity.")
            if time_of_day == 'Night':
                recommendations.append("Nighttime driving reduces visibility.")
            if vehicle_type == 'Motorcycle':
                recommendations.append("Motorcycles offer less protection in collisions.")

            st.write("### 🔍 Situational Analysis")
            if is_high_risk:
                st.write("The current combination of conditions highly correlates with severe or fatal accidents according to our models.")
                for rec in recommendations:
                    st.write(f"- ⚠️ {rec}")
                st.write("**Recommendation:** Avoid driving if possible, or reduce speed significantly.")
            else:
                st.write("Conditions are generally safe, but please exercise standard caution. Factors noted:")
                for rec in recommendations:
                    st.write(f"- Note: {rec}")
                if not recommendations:
                    st.write("- No extreme risk factors detected in the current environment.")

if __name__ == "__main__":
    main()
