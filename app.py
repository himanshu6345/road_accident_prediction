import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dynamic Data Mining Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS FOR STYLING ---
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    h1 { font-family: 'Inter', sans-serif; color: #2c3e50; text-align: center; margin-bottom: 0px !important; }
    .subtitle { text-align: center; color: #7f8c8d; font-size: 1.1em; margin-bottom: 40px; }
    .prediction-card { background-color: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 20px rgba(0,0,0,0.1); text-align: center; margin-top: 30px; }
    .pred-value { font-size: 2em; font-weight: bold; color: #2980b9; }
</style>
""", unsafe_allow_html=True)

# --- DYNAMIC TRAINING FUNCTION ---
def train_dynamic_model(df, target_col):
    st.session_state['training'] = True
    
    # Identify feature types
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    feature_info = {}
    
    # Impute missing values
    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
    if num_cols:
        num_imputer = SimpleImputer(strategy='median')
        X[num_cols] = num_imputer.fit_transform(X[num_cols])
        
    # Encode categorical features
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        feature_info[col] = {'type': 'categorical', 'options': list(le.classes_)}
        
    # Get info for numerical features
    for col in num_cols:
        feature_info[col] = {
            'type': 'numerical', 
            'min': float(X[col].min()), 
            'max': float(X[col].max()), 
            'mean': float(X[col].mean())
        }

    # Encode target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    
    # Scale numerical
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Models
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_scaled, y)
    
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_scaled, y)
    
    # Save to session state
    st.session_state['models'] = {
        'rf': rf_model,
        'svm': svm_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'feature_info': feature_info,
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'target_col': target_col
    }
    st.session_state['training'] = False
    return True

# --- DEFAULT MODEL LOADER ---
def load_default_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        rf_model = joblib.load(os.path.join(current_dir, 'rf_model.pkl'))
        svm_model = joblib.load(os.path.join(current_dir, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))
        label_encoders = joblib.load(os.path.join(current_dir, 'label_encoders.pkl'))
        target_encoder = joblib.load(os.path.join(current_dir, 'target_encoder.pkl'))
        
        # Hardcode feature info for default model
        feature_info = {
            'Weather_Condition': {'type': 'categorical', 'options': ['Clear', 'Rain', 'Snow', 'Fog']},
            'Road_Type': {'type': 'categorical', 'options': ['Highway', 'City Street', 'Rural Road']},
            'Speed_Limit': {'type': 'numerical', 'min': 15.0, 'max': 220.0, 'mean': 45.0},
            'Time_of_Day': {'type': 'categorical', 'options': ['Morning', 'Afternoon', 'Evening', 'Night']},
            'Vehicle_Type': {'type': 'categorical', 'options': ['Car', 'Truck', 'Motorcycle', 'Bus']},
            'Driver_Age': {'type': 'numerical', 'min': 16.0, 'max': 100.0, 'mean': 35.0}
        }
        
        return {
            'rf': rf_model, 'svm': svm_model, 'scaler': scaler,
            'label_encoders': label_encoders, 'target_encoder': target_encoder,
            'feature_info': feature_info,
            'cat_cols': ['Weather_Condition', 'Road_Type', 'Time_of_Day', 'Vehicle_Type'],
            'num_cols': ['Speed_Limit', 'Driver_Age'],
            'target_col': 'Accident_Severity'
        }
    except Exception as e:
        return None

# --- MAIN APP ---
def main():
    st.markdown("<h1>Dynamic Data Mining Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Upload any dataset to train and predict instantly!</div>", unsafe_allow_html=True)

    # Sidebar for Upload
    st.sidebar.header("📁 Upload Custom Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
            target_col = st.sidebar.selectbox("Select Target Column to Predict:", df.columns.tolist(), index=len(df.columns)-1)
            
            if st.sidebar.button("Train Models Now 🚀"):
                with st.spinner("Training models in the background..."):
                    train_dynamic_model(df, target_col)
                st.sidebar.success("Training Complete!")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            
    st.sidebar.markdown("---")
    st.sidebar.info("If no dataset is uploaded, the app will use the default Road Accident Prediction models.")

    # Load appropriate models into context
    if 'models' in st.session_state:
        context = st.session_state['models']
    else:
        context = load_default_models()
        
    if context is None:
        st.error("⚠️ No models available. Please upload a dataset or ensure default models exist.")
        return

    # Extract context
    feature_info = context['feature_info']
    target_col = context['target_col']
    rf_model = context['rf']
    svm_model = context['svm']
    scaler = context['scaler']
    label_encoders = context['label_encoders']
    target_encoder = context['target_encoder']
    cat_cols = context['cat_cols']
    num_cols = context['num_cols']

    st.write(f"### 📋 Enter Parameters (Predicting: **{target_col}**)")
    
    # Dynamic UI Generation
    user_inputs = {}
    
    # We create a 2-column layout
    cols = st.columns(2)
    for i, (feature_name, info) in enumerate(feature_info.items()):
        col = cols[i % 2]
        with col:
            if info['type'] == 'categorical':
                user_inputs[feature_name] = st.selectbox(feature_name, info['options'])
            else:
                user_inputs[feature_name] = st.number_input(
                    feature_name, 
                    min_value=info['min'], 
                    max_value=info['max'], 
                    value=info['mean'],
                    step=float((info['max'] - info['min']) / 20) if info['max'] != info['min'] else 1.0
                )

    # Prediction Logic
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Predict Outcome 🎯", use_container_width=True):
        with st.spinner('Analyzing...'):
            input_df = pd.DataFrame([user_inputs])
            
            # Encode categorical
            for c in cat_cols:
                le = label_encoders[c]
                # Fallback for unseen data
                if input_df[c][0] not in le.classes_:
                    input_df[c] = le.transform([le.classes_[0]])
                else:
                    input_df[c] = le.transform(input_df[c])
                    
            # Scale
            # Ensure column order matches training
            input_ordered = input_df[cat_cols + num_cols] if 'training' in st.session_state else input_df[list(feature_info.keys())]
            
            try:
                 input_scaled = scaler.transform(input_ordered)
            except ValueError:
                 # fallback order
                 input_scaled = scaler.transform(input_df)

            # Predict
            rf_pred_enc = rf_model.predict(input_scaled)[0]
            rf_label = target_encoder.inverse_transform([rf_pred_enc])[0]
            
            svm_pred_enc = svm_model.predict(input_scaled)[0]
            svm_label = target_encoder.inverse_transform([svm_pred_enc])[0]

            st.markdown(f"""
            <div class='prediction-card'>
                <h3 style='color: #34495e;'>Algorithm Predictions</h3>
                <p><b>Random Forest:</b> <span class='pred-value'>{rf_label}</span></p>
                <p><b>Support Vector Machine:</b> <span class='pred-value'>{svm_label}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probabilities Graph
            st.write("---")
            st.write("### 📈 Prediction Probabilities")
            
            # Use RF probabilities
            rf_probs = rf_model.predict_proba(input_scaled)[0]
            prob_df = pd.DataFrame({
                'Class': target_encoder.classes_,
                'Probability (%)': np.round(rf_probs * 100, 2)
            })
            prob_df.set_index('Class', inplace=True)
            
            st.line_chart(prob_df, height=300)

if __name__ == "__main__":
    main()
