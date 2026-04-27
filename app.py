import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
import plotly.express as px

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

# --- USER AUTHENTICATION & DATABASE ---
USERS_DB = "users.json"

def load_users():
    if os.path.exists(USERS_DB):
        with open(USERS_DB, "r") as f:
            return json.load(f)
    else:
        # Default users
        default_users = {
            "admin": "admin123",
            "guest": "guest"
        }
        with open(USERS_DB, "w") as f:
            json.dump(default_users, f)
        return default_users

def save_user(username, password):
    users = load_users()
    users[username] = password
    with open(USERS_DB, "w") as f:
        json.dump(users, f)

def check_password():
    """Returns `True` if the user had a correct password."""
    users = load_users()

    def password_entered():
        if (
            st.session_state.get("username") in users
            and st.session_state.get("password") == users[st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            if "password" in st.session_state:
                del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state or not st.session_state["password_correct"]:
        st.markdown("<br><br><h2 style='text-align: center;'>🔒 Authorized Access Only</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            tab1, tab2 = st.tabs(["Sign In", "Register"])
            
            with tab1:
                st.write("Please sign in to access the dashboard.")
                st.text_input("Username", key="username")
                st.text_input("Password", type="password", key="password")
                if st.button("Sign In", use_container_width=True):
                    password_entered()
                    st.rerun()
                
                if "password_correct" in st.session_state and not st.session_state["password_correct"]:
                    st.error("😕 Username or password incorrect")
                    
            with tab2:
                st.write("Create a new account.")
                new_user = st.text_input("Choose Username", key="new_username")
                new_pass = st.text_input("Choose Password", type="password", key="new_password")
                if st.button("Register Account", use_container_width=True):
                    if new_user in users:
                        st.error("⚠️ Username already exists! Choose another.")
                    elif new_user == "" or new_pass == "":
                        st.error("⚠️ Username and Password cannot be empty.")
                    else:
                        save_user(new_user, new_pass)
                        st.success("✅ Account created! Please switch to the Sign In tab to log in.")
        return False
    else:
        # Password correct.
        return True

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
    # Train Models
    rf_model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    rf_model.fit(X_scaled, y)
    
    # SVM with RBF kernel is extremely slow on large datasets (O(n^3)). 
    # We subsample to a maximum of 2000 rows to ensure fast loading times.
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    if len(X_scaled) > 2000:
        np.random.seed(42)
        indices = np.random.choice(len(X_scaled), 2000, replace=False)
        svm_model.fit(X_scaled[indices], y[indices])
    else:
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
            'Road_Condition': {'type': 'categorical', 'options': ['Normal', 'Wet', 'Potholes', 'Hill Area']},
            'Speed_Limit': {'type': 'numerical', 'min': 15.0, 'max': 220.0, 'mean': 45.0},
            'Time_of_Day': {'type': 'categorical', 'options': ['Morning', 'Afternoon', 'Evening', 'Night']},
            'Vehicle_Type': {'type': 'categorical', 'options': ['Car', 'Truck', 'Motorcycle', 'Bus']},
            'Driver_Age': {'type': 'numerical', 'min': 16.0, 'max': 100.0, 'mean': 35.0}
        }
        
        return {
            'rf': rf_model, 'svm': svm_model, 'scaler': scaler,
            'label_encoders': label_encoders, 'target_encoder': target_encoder,
            'feature_info': feature_info,
            'cat_cols': ['Weather_Condition', 'Road_Type', 'Road_Condition', 'Time_of_Day', 'Vehicle_Type'],
            'num_cols': ['Speed_Limit', 'Driver_Age'],
            'target_col': 'Accident_Severity'
        }
    except Exception as e:
        return None

# --- MAIN APP ---
def main():
    # Display normal sized top image centered
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image("/Users/himanshuprajapati/.gemini/antigravity/brain/b065bf0b-01c9-4753-b07a-fecb377c74d7/top_image_normal_1777292074823.png", use_container_width=True)
    st.markdown("<h1>Dynamic Data Mining Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Upload any dataset to train and predict instantly!</div>", unsafe_allow_html=True)

    # Sidebar for Upload
    st.sidebar.header("🚪 Session")
    st.sidebar.write(f"Logged in as: **{st.session_state.get('username', 'User')}**")
    if st.sidebar.button("Logout"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
        
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ AI Settings")
    
    # Auto-load key from environment or secrets if available
    env_key = os.getenv("GEMINI_API_KEY", "")
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except FileNotFoundError:
        secret_key = ""
        
    default_api_key = secret_key or env_key
    gemini_api_key = st.sidebar.text_input("Gemini API Key", value=default_api_key, type="password", help="Enter your Gemini API key to activate the live AI Chatbot.")
    
    st.sidebar.markdown("---")
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
            
            # --- SITUATIONAL ANALYSIS ---
            st.write("### 🔍 Situational Analysis")
            
            recommendations = []
            if 'Weather_Condition' in user_inputs and user_inputs['Weather_Condition'] in ['Snow', 'Fog', 'Rain']:
                recommendations.append(f"Visibility and traction are reduced due to {user_inputs['Weather_Condition']}.")
            if 'Speed_Limit' in user_inputs and user_inputs['Speed_Limit'] >= 80:
                recommendations.append(f"High speeds ({user_inputs['Speed_Limit']}) significantly increase accident severity.")
            if 'Time_of_Day' in user_inputs and user_inputs['Time_of_Day'] == 'Night':
                recommendations.append("Nighttime driving reduces visibility.")
            if 'Vehicle_Type' in user_inputs and user_inputs['Vehicle_Type'] == 'Motorcycle':
                recommendations.append("Motorcycles offer less protection in collisions.")
            if 'Road_Condition' in user_inputs and user_inputs['Road_Condition'] in ['Hill Area', 'Potholes']:
                recommendations.append(f"Road condition ({user_inputs['Road_Condition']}) introduces severe handling risks.")

            # Generic risk check (works for default target classes and common binary classes)
            high_risk_labels = ['Severe', 'Fatal', 'High', 'Yes', 1, '1', 'True', True]
            is_high_risk = (rf_label in high_risk_labels) or (svm_label in high_risk_labels)
            
            if is_high_risk:
                st.error("🚨 **RISK ALERT: HIGH SEVERITY PREDICTED!**")
                if recommendations:
                    st.write("The current combination of conditions correlates with high-risk accidents. Factors noted:")
                    for rec in recommendations:
                        st.write(f"- ⚠️ {rec}")
            else:
                st.success("✅ **DRIVE SAFE: CONDITIONS ARE FAVORABLE!**")
                if recommendations:
                    st.write("Conditions are generally safe, but please exercise standard caution. Factors noted:")
                    for rec in recommendations:
                        st.write(f"- Note: {rec}")
                else:
                    st.write("- No extreme risk factors detected.")
            
            # --- OUTCOME PROBABILITIES ---
            st.write("---")
            st.write("### 📊 Specific Outcome Risks")
            
            # Use RF probabilities to calculate specific risks
            rf_probs = rf_model.predict_proba(input_scaled)[0]
            classes = list(target_encoder.classes_)
            
            p_fatal = rf_probs[classes.index('Fatal')] if 'Fatal' in classes else 0
            p_severe = rf_probs[classes.index('Severe')] if 'Severe' in classes else 0
            p_moderate = rf_probs[classes.index('Moderate')] if 'Moderate' in classes else 0
            p_minor = rf_probs[classes.index('Minor')] if 'Minor' in classes else 0
            
            # Calculate independent chances (out of 100%)
            chance_death = p_fatal * 100
            chance_injury = min((p_severe + (0.7 * p_moderate)) * 100, 100.0)
            chance_damage = min((p_minor + p_moderate + (0.5 * p_severe)) * 100, 100.0)
            
            # Create 3 columns for individual metric visualiztion
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"<h4 style='text-align: center; color: #c0392b;'>💀 Chances of Death</h4>", unsafe_allow_html=True)
                st.progress(int(chance_death))
                st.markdown(f"<h2 style='text-align: center;'>{chance_death:.1f}%</h2>", unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"<h4 style='text-align: center; color: #e67e22;'>🚑 Chances of Injuries</h4>", unsafe_allow_html=True)
                st.progress(int(chance_injury))
                st.markdown(f"<h2 style='text-align: center;'>{chance_injury:.1f}%</h2>", unsafe_allow_html=True)
                
            with col3:
                st.markdown(f"<h4 style='text-align: center; color: #f1c40f;'>💥 Property Damages</h4>", unsafe_allow_html=True)
                st.progress(int(chance_damage))
                st.markdown(f"<h2 style='text-align: center;'>{chance_damage:.1f}%</h2>", unsafe_allow_html=True)
                
            # Keep the original full probability graph as an expander for nerds
            with st.expander("View Full Severity Probability Breakdown"):
                prob_df = pd.DataFrame({
                    'Severity Level': classes,
                    'Probability (%)': np.round(rf_probs * 100, 2)
                })
                prob_df.set_index('Severity Level', inplace=True)
                st.bar_chart(prob_df, height=250)


    # --- 3D DATA VISUALIZATION ---
    st.write("---")
    st.write("### 🌌 3D Data Universe")
    st.write("Explore the multi-dimensional relationships of your dataset in this interactive 3D space.")
    
    try:
        if uploaded_file is not None:
            uploaded_file.seek(0)
            full_df = pd.read_csv(uploaded_file)
            plot_df = full_df.sample(min(1000, len(full_df)))
            cols = plot_df.columns.tolist()
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            full_df = pd.read_csv(os.path.join(current_dir, 'accident_data.csv'))
            plot_df = full_df.sample(min(1000, len(full_df)))
            cols = plot_df.columns.tolist()
            
        if len(cols) >= 3:
            # Try to pick interesting columns for X, Y, Z
            x_col = num_cols[0] if len(num_cols) > 0 else cols[0]
            y_col = num_cols[1] if len(num_cols) > 1 else cols[1]
            z_col = cat_cols[0] if len(cat_cols) > 0 else cols[2]
            
            fig = px.scatter_3d(
                plot_df, 
                x=x_col, 
                y=y_col, 
                z=z_col, 
                color=target_col if target_col in cols else cols[-1],
                opacity=0.7,
                title="3D Feature Distribution"
            )
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Dataset requires at least 3 columns for 3D visualization.")
    except Exception as e:
        st.warning("Could not generate 3D graph for the current dataset.")

    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image("/Users/himanshuprajapati/.gemini/antigravity/brain/b065bf0b-01c9-4753-b07a-fecb377c74d7/bottom_image_normal_1777292095383.png", use_container_width=True)

    # --- AI CHATBOT INTERFACE ---
    st.markdown("---")
    st.header("💬 AI Data Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I am your AI Data Assistant. I can help explain the Random Forest and SVM predictions, or answer questions about your data. How can I help you today?"})

    # Configure Gemini if key is provided
    model = None
    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            st.error(f"Error configuring Gemini API: {e}")
    else:
        st.info("💡 **Tip**: Enter your Gemini API Key in the sidebar to activate the live AI Chatbot! Running in offline mock mode for now.")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # --- AI RESPONSE GENERATOR ---
        if model:
            # LIVE GEMINI RESPONSE
            try:
                with st.spinner("Thinking..."):
                    # Build context prompt
                    system_context = f"You are a helpful AI Data Science assistant embedded in a Road Accident Prediction Streamlit app. The user is currently predicting the column '{target_col}'. The input features and their metadata are: {feature_info}. Keep your answers concise, formatted in markdown, and directly relevant to their dataset and Random Forest/SVM models."
                    
                    history_str = "\\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages])
                    full_prompt = f"{system_context}\\n\\nHere is the conversation history:\\n{history_str}\\n\\nPlease respond to the user's latest query."
                    
                    response = model.generate_content(full_prompt)
                    bot_response = response.text
            except Exception as e:
                bot_response = f"Failed to generate response: {e}"
        else:
            # OWN CUSTOM LOCAL DATA AGENT
            lower_prompt = prompt.lower()
            
            # Simple Math & Data Queries
            if "speed limit" in lower_prompt and "average" in lower_prompt:
                if 'Speed_Limit' in full_df.columns:
                    bot_response = f"📊 The average speed limit across the entire dataset is **{full_df['Speed_Limit'].mean():.2f} mph**."
                else: bot_response = "I couldn't find the 'Speed_Limit' column in your dataset."
            elif "speed limit" in lower_prompt and ("highest" in lower_prompt or "max" in lower_prompt):
                if 'Speed_Limit' in full_df.columns:
                    bot_response = f"📊 The highest speed limit recorded in the dataset is **{full_df['Speed_Limit'].max()} mph**."
                else: bot_response = "I couldn't find the 'Speed_Limit' column in your dataset."
            elif "fatal" in lower_prompt and ("how many" in lower_prompt or "count" in lower_prompt):
                if 'Accident_Severity' in full_df.columns:
                    count = len(full_df[full_df['Accident_Severity'] == 'Fatal'])
                    bot_response = f"⚠️ There are exactly **{count}** fatal accidents recorded in your dataset."
                else: bot_response = "I couldn't find the 'Accident_Severity' column to count fatalities."
            elif "rain" in lower_prompt and ("how many" in lower_prompt or "count" in lower_prompt):
                if 'Weather_Conditions' in full_df.columns:
                    count = len(full_df[full_df['Weather_Conditions'] == 'Raining'])
                    bot_response = f"🌧️ There are **{count}** accidents that occurred during rainy weather conditions."
                else: bot_response = "I couldn't find the 'Weather_Conditions' column."
            elif "row" in lower_prompt or "size" in lower_prompt or "how big" in lower_prompt:
                bot_response = f"📈 Your dataset currently contains **{len(full_df)} rows** and **{len(full_df.columns)} columns**."
            elif "fatal" in lower_prompt or "severe" in lower_prompt:
                bot_response = "The models typically predict 'Fatal' or 'Severe' when high-risk factors align. For example, driving at extreme speeds, during bad weather, or on a Motorcycle drastically increases the severity score."
            elif "random forest" in lower_prompt or "rf" in lower_prompt:
                bot_response = "Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the most popular class. It's very robust against overfitting!"
            elif "svm" in lower_prompt or "support vector" in lower_prompt:
                bot_response = "Support Vector Machine (SVM) finds the optimal boundary that separates our different severity classes. We are using an 'RBF' kernel to handle complex, non-linear relationships in the dataset."
            elif "dataset" in lower_prompt or "data" in lower_prompt:
                bot_response = f"Your current dataset is using {len(feature_info)} input features to predict the '{target_col}' column."
            else:
                bot_response = "🤖 **I am your Custom Data Agent!** Since you haven't provided a Gemini API Key, I am analyzing your dataset locally.\\n\\nTry asking me specific math questions like:\\n- *'What is the average speed limit?'*\\n- *'How many fatal accidents are there?'*\\n- *'How many rows are in the dataset?'*"
            
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

if __name__ == "__main__":
    if check_password():
        main()
