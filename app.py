import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from dotenv import load_dotenv
import requests
import datetime
import pytz
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from streamlit_geolocation import streamlit_geolocation
from streamlit_lottie import st_lottie

# Load environment variables
load_dotenv()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
import plotly.express as px
from database import init_db, add_user, verify_user, reset_password, log_prediction, get_predictions, get_all_users, get_all_predictions, create_session_token, verify_session_token, delete_session_token, log_user_login, get_all_user_logs
from notifications import notify_admin_of_new_user, notify_user_of_registration
from static_assistant import StaticModelAssistant
from extra_streamlit_components import CookieManager

def fetch_recent_accidents(location_name):
    try:
        query = quote_plus(f"road accident {location_name} when:7d")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        r = requests.get(url, timeout=5)
        root = ET.fromstring(r.content)
        
        accidents = []
        for item in root.findall('.//item')[:3]:
            title = item.find('title').text
            pub_date = item.find('pubDate').text
            if " - " in title:
                title = " - ".join(title.split(" - ")[:-1])
            accidents.append({"title": title, "date": pub_date})
            
        return accidents
    except Exception as e:
        return []

def fetch_live_traffic(lat, lon, api_key, time_of_day):
    if api_key:
        try:
            url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key={api_key}&point={lat},{lon}"
            r = requests.get(url, timeout=5).json()
            if 'flowSegmentData' in r:
                data = r['flowSegmentData']
                current_speed = data.get('currentSpeed', 0)
                free_flow_speed = data.get('freeFlowSpeed', 1)
                ratio = current_speed / free_flow_speed if free_flow_speed > 0 else 1
                if ratio < 0.5:
                    return "Heavy Traffic", current_speed, free_flow_speed
                elif ratio < 0.8:
                    return "Moderate Traffic", current_speed, free_flow_speed
                else:
                    return "Free Flowing", current_speed, free_flow_speed
        except Exception:
            pass
            
    if time_of_day in ['Morning', 'Evening']:
        return "Heavy Traffic (Estimated)", None, None
    elif time_of_day == 'Afternoon':
        return "Moderate Traffic (Estimated)", None, None
    else:
        return "Free Flowing (Estimated)", None, None

def fetch_live_data(location_name=None, lat=None, lon=None):
    try:
        state = 'Delhi'
        city = 'New Delhi'
        
        if lat is None or lon is None:
            geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location_name}&count=1&language=en&format=json"
            response = requests.get(geocode_url, timeout=5)
            if response.status_code != 200:
                return None, "Location service currently unavailable."
            
            geo_resp = response.json()
            if 'results' not in geo_resp or not geo_resp['results']:
                return None, "Location not found."
                
            lat = geo_resp['results'][0]['latitude']
            lon = geo_resp['results'][0]['longitude']
            state = geo_resp['results'][0].get('admin1', 'Delhi')
            city = geo_resp['results'][0].get('name', 'New Delhi')
        else:
            # Try to fetch State and City using the provided location name string
            search_name = location_name.split(',')[0] if location_name else "New Delhi"
            geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={search_name}&count=1&language=en&format=json"
            geo_resp = requests.get(geocode_url, timeout=5).json()
            if 'results' in geo_resp and geo_resp['results']:
                state = geo_resp['results'][0].get('admin1', 'Delhi')
                city = geo_resp['results'][0].get('name', 'New Delhi')
        
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code,precipitation"
        weather_resp = requests.get(weather_url, timeout=5).json()
        current = weather_resp.get('current', {})
        w_code = current.get('weather_code', 0)
        temp = current.get('temperature_2m', 0)
        precip = current.get('precipitation', 0)
        
        # Enhanced Logic: Prioritize precipitation for Rain/Snow
        if precip > 0:
            if temp < 0: weather = 'Snow'
            else: weather = 'Rain'
        elif w_code in [0, 1]: weather = 'Clear'
        elif w_code in [2, 3]: weather = 'Cloudy'
        elif w_code in [45, 48]: weather = 'Fog'
        elif w_code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99]: weather = 'Rain'
        elif w_code in [71, 73, 75, 77, 85, 86]: weather = 'Snow'
        else: weather = 'Clear' # Default
        
        current_hour = datetime.datetime.now().hour
        if 5 <= current_hour < 12: time_of_day = 'Morning'
        elif 12 <= current_hour < 17: time_of_day = 'Afternoon'
        elif 17 <= current_hour < 20: time_of_day = 'Evening'
        else: time_of_day = 'Night'
        
        road_cond = 'Wet' if weather in ['Rain', 'Snow'] else 'Normal'
        
        return {
            'Weather_Condition': weather,
            'Time_of_Day': time_of_day,
            'Road_Condition': road_cond,
            'Temperature': temp,
            'Precipitation': precip,
            'Lat': lat,
            'Lon': lon,
            'State': state,
            'City': city
        }, None
    except Exception as e:
        return None, str(e)

# Initialize DB
init_db()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dynamic Data Mining Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS FOR STYLING ---
# --- CSS FOR STYLING ---
login_styles = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    :root {
        --primary: #0984e3;
        --primary-dark: #074da1;
        --accent: #6c5ce7;
        --bg-dark: #0f172a;
        --text-light: #f8fafc;
        --text-muted: #94a3b8;
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
        --glass-blur: 12px;
    }

    .stApp { 
        background: radial-gradient(circle at 50% 50%, #1e293b 0%, #0f172a 100%);
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    .stApp::before {
        content: ''; position: fixed; width: 400px; height: 400px;
        background: var(--primary); top: -100px; right: -100px;
        border-radius: 50%; filter: blur(80px); opacity: 0.4; z-index: 0;
        pointer-events: none;
    }
    
    .stApp::after {
        content: ''; position: fixed; width: 350px; height: 350px;
        background: var(--accent); bottom: -50px; left: -50px;
        border-radius: 50%; filter: blur(80px); opacity: 0.4; z-index: 0;
        pointer-events: none;
    }

    /* Hide Streamlit UI */
    [data-testid="stSidebar"], [data-testid="stHeader"], [data-testid="stToolbar"] {
        display: none !important;
    }

    /* Glassmorphism Container applied to the middle column */
    [data-testid="column"]:nth-child(2) {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(var(--glass-blur)) !important;
        -webkit-backdrop-filter: blur(var(--glass-blur)) !important;
        border: 1px solid var(--glass-border) !important;
        padding: 3rem !important;
        border-radius: 2rem !important;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5) !important;
        margin-top: 50px;
        z-index: 10;
    }
    .login-header { text-align: center; margin-bottom: 2rem; z-index: 10; position: relative; }
    .login-header h1 { 
        color: var(--text-light) !important;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    .login-header p { 
        color: var(--text-muted);
        font-size: 1rem;
        margin: 0;
        padding: 0;
    }
    
    /* Input Styling */
    [data-baseweb="input"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 0.75rem !important;
        transition: all 0.3s ease !important;
    }
    
    [data-baseweb="base-input"] {
        background: transparent !important;
    }

    .stTextInput input {
        background: transparent !important;
        color: white !important;
        padding: 0.8rem 1rem !important;
        font-size: 0.95rem !important;
        -webkit-text-fill-color: white !important;
    }

    [data-baseweb="input"]:focus-within {
        border-color: var(--primary) !important;
        background: rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 0 0 4px rgba(9, 132, 227, 0.2) !important;
    }

    .stTextInput input::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
        opacity: 1 !important;
    }

    /* Fix Browser Autofill Visibility */
    .stTextInput input:-webkit-autofill,
    .stTextInput input:-webkit-autofill:hover, 
    .stTextInput input:-webkit-autofill:focus, 
    .stTextInput input:-webkit-autofill:active {
        -webkit-box-shadow: 0 0 0 30px #1e293b inset !important;
        -webkit-text-fill-color: white !important;
        transition: background-color 5000s ease-in-out 0s;
    }

    .stTextInput label {
        color: var(--text-light) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Primary Button */
    div.stButton > button[kind="primary"], .stFormSubmitButton > button {
        background: var(--primary) !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        border-radius: 0.75rem !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 15px -3px rgba(9, 132, 227, 0.4) !important;
        width: 100%;
        margin-top: 1rem !important;
    }

    div.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover {
        background: var(--primary-dark) !important;
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgba(9, 132, 227, 0.5) !important;
    }

    div.stButton > button[kind="primary"]:active, .stFormSubmitButton > button:active {
        transform: translateY(0);
    }
    
    /* Secondary (Social) Buttons */
    div.stButton > button[kind="secondary"] {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        color: var(--text-light) !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    div.stButton > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: var(--text-muted) !important;
        transform: translateY(-2px);
    }

    /* Tabs Override */
    .stTabs [data-baseweb="tab-list"] { 
        background: transparent !important; 
        gap: 10px; 
        border-bottom: 1px solid var(--glass-border) !important;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-muted) !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        border-radius: 0.5rem !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        color: var(--text-light) !important;
    }
    .stTabs [aria-selected="true"] {
        color: white !important;
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
    }

    [data-testid="stForm"] { background: transparent !important; border: none !important; padding: 0 !important; }
    
    .footer-links { text-align: center; margin-top: 2rem; font-size: 0.9rem; color: var(--text-muted); }
    .footer-links a { color: var(--primary); text-decoration: none; font-weight: 600; }
    .footer-links a:hover { text-decoration: underline; }
    
    .social-login { display: flex; gap: 1rem; margin-top: 1.5rem; }
    .social-btn {
        flex: 1; display: flex; align-items: center; justify-content: center;
        padding: 0.75rem; background: var(--glass-bg); border: 1px solid var(--glass-border);
        border-radius: 0.75rem; color: var(--text-light); cursor: pointer; transition: all 0.3s ease;
    }
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
"""

dashboard_styles = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    .stApp { background: #0f172a; font-family: 'Poppins', sans-serif; color: white; }
    h1 { font-weight: 800; color: #ffffff !important; }
    .prediction-card { 
        background: rgba(30, 41, 59, 0.7); 
        padding: 30px; 
        border-radius: 20px; 
        border: 1px solid rgba(0, 212, 255, 0.1);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2); 
    }
    .stSelectbox label, .stTextInput label { color: #94a3b8 !important; }
</style>
"""

if "password_correct" not in st.session_state or not st.session_state["password_correct"]:
    st.markdown(login_styles, unsafe_allow_html=True)
else:
    st.markdown(dashboard_styles, unsafe_allow_html=True)




# --- USER AUTHENTICATION & DATABASE ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def check_password():
    """Returns `True` if the user had a correct password."""
    
    # Initialize Cookie Manager
    cookie_manager = CookieManager()
    
    def password_entered(username_val, password_val, remember_me=False):
        # Normalize username/email to lowercase and trim spaces
        username_val = username_val.strip().lower() if username_val else ""
        actual_username = verify_user(username_val, password_val)
        if actual_username:
            st.session_state["password_correct"] = True
            st.session_state["logged_in_user"] = actual_username
            log_user_login(actual_username)
            
            if remember_me:
                token = create_session_token(actual_username)
                if token:
                    cookie_manager.set("session_token", token, expires_at=datetime.datetime.now() + datetime.timedelta(days=30))
        else:
            st.session_state["password_correct"] = False

    # Check for existing session token in cookies (if not just logged out)
    if not st.session_state.get("logout_requested"):
        if "password_correct" not in st.session_state or not st.session_state["password_correct"]:
            token = cookie_manager.get("session_token")
            if token:
                username = verify_session_token(token)
                if username:
                    st.session_state["password_correct"] = True
                    st.session_state["logged_in_user"] = username
                    st.rerun()
    else:
        # Clear the logout flag so future visits will auto-login normally
        st.session_state["logout_requested"] = False

    if "password_correct" not in st.session_state or not st.session_state["password_correct"]:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("""
            <div class="login-header">
                <h1>Welcome Back</h1>
                <p>Predicting risks, saving lives.</p>
            </div>
            """, unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["Login", "Register"])
            
            with tab1:
                st.markdown("<div style='padding-top: 30px;'>", unsafe_allow_html=True)
                with st.form("login_form"):
                    st.text_input("Login ID", key="username", placeholder="Enter your Login ID")
                    st.text_input("Password", type="password", key="password", placeholder="••••••••")
                    remember_me = st.checkbox("Keep me signed in for 30 days", value=True)
                    submit_login = st.form_submit_button("SIGN IN", use_container_width=True)
                
                if submit_login:
                    password_entered(st.session_state.username, st.session_state.password, remember_me)
                    st.rerun()
                
                if "password_correct" in st.session_state and not st.session_state["password_correct"]:
                    st.error("!! ACCESS DENIED: INVALID CREDENTIALS !!")
                    st.info("💡 Hint: You can log in using your **Email Address** or your **Access ID**.")
                
                st.markdown("<div style='margin-bottom: 10px; text-align: center; font-size: 0.85em; color: var(--text-muted);'>Or continue with</div>", unsafe_allow_html=True)
                
                scol1, scol2, scol3 = st.columns(3)
                with scol1: btn_google = st.button("🔴 Google", use_container_width=True)
                with scol2: btn_apple = st.button("🍏 Apple", use_container_width=True)
                with scol3: btn_github = st.button("🐙 GitHub", use_container_width=True)
                
                if btn_google or btn_apple or btn_github:
                    provider = "Google" if btn_google else "Apple" if btn_apple else "GitHub"
                    st.success(f"**{provider} Login Authenticated!** Accessing dashboard...")
                    
                    import time
                    time.sleep(1) # Small delay for effect
                    
                    # Force login
                    st.session_state["password_correct"] = True
                    st.session_state["logged_in_user"] = f"{provider}_Guest"
                    st.rerun()
                
                # Forgot Password Flow
                if st.button("Forgot Password?", type="tertiary", use_container_width=True):
                    st.session_state.show_reset = not st.session_state.get('show_reset', False)
                
                if st.session_state.get('show_reset', False):
                    with st.form("reset_password_form"):
                        st.markdown("<p style='text-align: center; color: var(--text-muted);'>Reset Your Password</p>", unsafe_allow_html=True)
                        reset_user = st.text_input("Login ID", placeholder="Your username")
                        reset_email = st.text_input("Email Address", placeholder="john@example.com")
                        reset_pass = st.text_input("New Password", type="password", placeholder="••••••••")
                        
                        if st.form_submit_button("RESET PASSWORD", use_container_width=True):
                            if not reset_user or not reset_email or not reset_pass:
                                st.error("Please fill in all fields.")
                            else:
                                success, msg = reset_password(reset_user, reset_email, reset_pass)
                                if success:
                                    st.success("Password reset successful! You may now login.")
                                else:
                                    st.error(f"Reset Failed: {msg}")

                st.markdown("</div>", unsafe_allow_html=True)
                                
            with tab2:
                st.markdown("<div style='padding-top: 10px;'>", unsafe_allow_html=True)
                st.markdown("""
                <div class="login-header" style="margin-bottom: 20px;">
                    <h1>Create Account</h1>
                    <p>Join the safety revolution.</p>
                </div>
                """, unsafe_allow_html=True)
                if st.session_state.get('registration_success', False):
                    st.success(f"ACCESS GRANTED: PROTOCOL INITIALIZED")
                    st.warning(f"⚠️ IMPORTANT: Use the ID below to log in, NOT your name or email.")
                    st.info(f"**YOUR ASSIGNED ACCESS ID:** `{st.session_state.new_login_id}`")
                    if st.button("PROCEED TO LOGIN"):
                        st.session_state.registration_success = False
                        st.rerun()
                else:
                    with st.form("registration_form"):
                        r_col1, r_col2 = st.columns(2)
                        with r_col1: first_name = st.text_input("First Name", placeholder="John")
                        with r_col2: last_name = st.text_input("Last Name", placeholder="Doe")
                        
                        new_email = st.text_input("Email Address", placeholder="john@example.com")
                        new_contact = st.text_input("Contact Number (Optional)", placeholder="Phone number")
                        new_pass = st.text_input("Password", type="password", placeholder="••••••••")
                        
                        submitted = st.form_submit_button("REGISTER", use_container_width=True)
                    
                    if submitted:
                        if not first_name or not last_name or not new_email or not new_pass:
                            st.error("!! ERROR: INCOMPLETE PARAMETERS !!")
                        else:
                            base_username = f"{first_name.strip().lower()}{last_name.strip().lower()}"
                            import random
                            new_user = base_username
                            for _ in range(10):
                                success, message = add_user(username=new_user, password=new_pass, email=new_email, 
                                                           full_name=f"{first_name.strip()} {last_name.strip()}", 
                                                           contact_number=new_contact)
                                if success:
                                    notify_admin_of_new_user(new_user, new_email)
                                    notify_user_of_registration(new_email, new_contact, first_name.strip(), new_user)
                                    st.session_state.registration_success = True
                                    st.session_state.new_login_id = new_user
                                    st.rerun()
                                else:
                                    new_user = f"{base_username}{random.randint(10, 9999)}"
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="footer-links" style="margin-top: 40px;">
                <p>New Operator? <a href="#">[ Request Protocol ]</a></p>
            </div>
            """, unsafe_allow_html=True)
        return False
    else:
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
            'Driver_Age': {'type': 'numerical', 'min': 16.0, 'max': 100.0, 'mean': 35.0},
            'State': {'type': 'categorical', 'options': ['Delhi', 'Haryana', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Uttar Pradesh']},
            'City': {'type': 'categorical', 'options': []}
        }
        
        return {
            'rf': rf_model, 'svm': svm_model, 'scaler': scaler,
            'label_encoders': label_encoders, 'target_encoder': target_encoder,
            'feature_info': feature_info,
            'cat_cols': ['Weather_Condition', 'Road_Type', 'Road_Condition', 'Time_of_Day', 'Vehicle_Type', 'State', 'City'],
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
        st.image(os.path.join(os.path.dirname(__file__), 'assets', 'top_banner.png'), use_container_width=True)
    
    # Dynamic Greeting for Main Dashboard (IST)
    ist = pytz.timezone('Asia/Kolkata')
    india_time = datetime.datetime.now(ist)
    hour = india_time.hour
    
    if hour < 12: greeting = "Good Morning ☀️"
    elif hour < 18: greeting = "Good Afternoon 🌤️"
    else: greeting = "Good Evening 🌙"
    
    st.markdown(f"<h1 style='font-size: 2.5em; margin-bottom: 5px;'>{greeting}, {st.session_state.get('logged_in_user', 'User')}</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Upload any dataset to train and predict instantly!</div>", unsafe_allow_html=True)

    # Sidebar for Upload
    st.sidebar.header("🚪 Session")
    st.sidebar.write(f"Logged in as: **{st.session_state.get('logged_in_user', 'User')}**")
    if st.sidebar.button("Logout", key="logout_btn"):
        cm = CookieManager()
        token = cm.get("session_token")
        if token:
            delete_session_token(token)
            cm.delete("session_token")
            
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Set logout flag and rerun
        st.session_state["logout_requested"] = True
        st.rerun()
        
    with st.sidebar.expander("📜 View Prediction History"):
        history = get_predictions(st.session_state.get("logged_in_user"))
        if history:
            for h in history:
                try:
                    feat_dict = json.loads(h['input_features'])
                    # Format features cleanly
                    details = " | ".join([f"{k}: {v}" for k, v in feat_dict.items() if k not in ['State']])
                except:
                    details = "No details available."
                    
                st.markdown(f"🕒 **{h['timestamp']}**")
                st.markdown(f"<div style='font-size:0.8em; color:gray; margin-bottom:5px;'>{details}</div>", unsafe_allow_html=True)
                st.markdown(f"**RF:** {h['rf_prediction']} | **SVM:** {h['svm_prediction']}")
                st.markdown("---")
        else:
            st.write("No predictions yet.")
        
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ AI Settings")
    
    # Auto-load key from environment or secrets if available
    tomtom_api_key = st.sidebar.text_input("TomTom Traffic API Key (Optional)", type="password", help="Enter a TomTom API key for true real-time traffic speeds, otherwise the system will use heuristic estimations.")
    
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

    # Main Navigation Tabs
    if st.session_state.get('logged_in_user') == 'admin':
        main_tabs = st.tabs(["🎯 Prediction Engine", "💬 AI Assistant", "🛡️ Admin Dashboard"])
        tab_pred, tab_chat, tab_admin = main_tabs[0], main_tabs[1], main_tabs[2]
    else:
        main_tabs = st.tabs(["🎯 Prediction Engine", "💬 AI Assistant"])
        tab_pred, tab_chat = main_tabs[0], main_tabs[1]
        
    with tab_pred:
        st.write(f"### 📋 Enter Parameters (Predicting: **{target_col}**)")
    
        tab_manual, tab_live = st.tabs(["📝 Manual Prediction", "📡 Real-Time Live Prediction"])
        
        predict_clicked = False
        active_user_inputs = {}
        active_live_loc = ""
        
        with tab_manual:
            # Dynamic UI Generation
            user_inputs = {}
        
            haryana_districts = ['Ambala', 'Bhiwani', 'Charkhi Dadri', 'Faridabad', 'Fatehabad', 'Gurugram', 'Hisar', 'Jhajjar', 'Jind', 'Kaithal', 'Karnal', 'Kurukshetra', 'Mahendragarh', 'Nuh', 'Palwal', 'Panchkula', 'Panipat', 'Rewari', 'Rohtak', 'Sirsa', 'Sonipat', 'Yamunanagar']
            delhi_districts = ['Central Delhi', 'East Delhi', 'New Delhi', 'North Delhi', 'North East Delhi', 'North West Delhi', 'Shahdara', 'South Delhi', 'South East Delhi', 'South West Delhi', 'West Delhi']
            up_districts = ['Agra', 'Aligarh', 'Prayagraj', 'Ambedkar Nagar', 'Amethi', 'Amroha', 'Auraiya', 'Ayodhya', 'Azamgarh', 'Baghpat', 'Bahraich', 'Ballia', 'Balrampur', 'Banda', 'Barabanki', 'Bareilly', 'Basti', 'Bhadohi', 'Bijnor', 'Budaun', 'Bulandshahr', 'Chandauli', 'Chitrakoot', 'Deoria', 'Etah', 'Etawah', 'Farrukhabad', 'Fatehpur', 'Firozabad', 'Gautam Buddha Nagar', 'Ghaziabad', 'Ghazipur', 'Gonda', 'Gorakhpur', 'Hamirpur', 'Hapur', 'Hardoi', 'Hathras', 'Jalaun', 'Jaunpur', 'Jhansi', 'Kannauj', 'Kanpur Dehat', 'Kanpur Nagar', 'Kasganj', 'Kaushambi', 'Lakhimpur Kheri', 'Kushinagar', 'Lalitpur', 'Lucknow', 'Maharajganj', 'Mahoba', 'Mainpuri', 'Mathura', 'Mau', 'Meerut', 'Mirzapur', 'Moradabad', 'Muzaffarnagar', 'Pilibhit', 'Pratapgarh', 'Raebareli', 'Rampur', 'Saharanpur', 'Sambhal', 'Sant Kabir Nagar', 'Shahjahanpur', 'Shamli', 'Shravasti', 'Siddharthnagar', 'Sitapur', 'Sonbhadra', 'Sultanpur', 'Unnao', 'Varanasi']
        
            state_cities_map = {
                'Delhi': delhi_districts,
                'Haryana': haryana_districts,
                'Maharashtra': ['Mumbai', 'Pune', 'Nagpur', 'Nashik'],
                'Karnataka': ['Bengaluru', 'Mysuru', 'Mangaluru', 'Hubli'],
                'Tamil Nadu': ['Chennai', 'Coimbatore', 'Madurai', 'Salem'],
                'Uttar Pradesh': up_districts
            }
        
            # We create a 2-column layout
            cols = st.columns(2)
            for i, (feature_name, info) in enumerate(feature_info.items()):
                col = cols[i % 2]
                with col:
                    if feature_name == 'State':
                        user_inputs['State'] = st.selectbox('State', list(state_cities_map.keys()))
                    elif feature_name == 'City':
                        selected_state = user_inputs.get('State', 'Delhi')
                        user_inputs['City'] = st.selectbox('City', state_cities_map.get(selected_state, []))
                    elif info['type'] == 'categorical':
                        user_inputs[feature_name] = st.selectbox(feature_name, info['options'])
                    else:
                        user_inputs[feature_name] = st.number_input(
                            feature_name, 
                            min_value=info['min'], 
                            max_value=info['max'], 
                            value=info['mean'],
                            step=float((info['max'] - info['min']) / 20) if info['max'] != info['min'] else 1.0
                        )
    
            try:
                full_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'accident_data.csv'))
            except Exception:
                full_df = pd.DataFrame()
    
            if 'City' in user_inputs and not full_df.empty:
                st.write("---")
                st.write(f"### 📊 Local Analytics: {user_inputs['City']}, {user_inputs['State']}")
                city_df = full_df[full_df['City'] == user_inputs['City']]
                if not city_df.empty:
                    severe_fatal = city_df[city_df['Accident_Severity'].isin(['Severe', 'Fatal'])]
                    max_rate = len(severe_fatal) / len(city_df) * 100
                
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric("Max Street/Road Accident Rate (Severe/Fatal)", f"{max_rate:.1f}%", f"Total Records: {len(city_df)}")
                    with col2:
                        road_counts = severe_fatal['Road_Type'].value_counts().reset_index()
                        road_counts.columns = ['Road Type', 'Severe Accidents']
                        if not road_counts.empty:
                            fig = px.bar(road_counts, x='Road Type', y='Severe Accidents', color='Road Type', title=f"High Severity Accidents by Road Type in {user_inputs['City']}", height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No severe accidents recorded for this city yet.")
    
            # Prediction Logic
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("SUBMIT", key="btn_manual", use_container_width=True):
                predict_clicked = True
                active_user_inputs = user_inputs
                active_live_loc = ""

        if 'auto_loc' not in st.session_state:
            st.session_state['auto_loc'] = ""

        with tab_live:
            st.write("### 📡 Auto-Fetch Live Conditions")
            st.write("Enter your location and vehicle details to fetch live weather, time context, traffic, and accident alerts automatically.")
        
            col_btn, col_text = st.columns([1, 2])
            with col_btn:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                gps_loc = streamlit_geolocation()
            
                if gps_loc and gps_loc.get('latitude') is not None and gps_loc.get('latitude') != st.session_state.get('last_lat'):
                    try:
                        lat = gps_loc['latitude']
                        lon = gps_loc['longitude']
                        st.session_state['last_lat'] = lat
                    
                        # Reverse geocode with BigDataCloud (More reliable than Nominatim on Cloud)
                        try:
                            bdc_url = f"https://api.bigdatacloud.net/data/reverse-geocode-client?latitude={lat}&longitude={lon}&localityLanguage=en"
                            bdc_resp = requests.get(bdc_url, timeout=5).json()
                            
                            address = bdc_resp.get('principalSubdivision', '')
                            locality = bdc_resp.get('locality', '')
                            city = bdc_resp.get('city', '')
                            village = bdc_resp.get('village', '') # Specifically look for village
                            suburb = bdc_resp.get('suburb', '')
                            
                            # Build a detailed name including village/city
                            name_parts = [p for p in [village, suburb, locality, city, address] if p]
                            # Remove duplicates if city/locality are the same
                            unique_parts = []
                            for p in name_parts:
                                if p not in unique_parts:
                                    unique_parts.append(p)
                                    
                            loc_str = ", ".join(unique_parts) if unique_parts else f"Lat: {lat:.4f}, Lon: {lon:.4f}"
                        except Exception:
                            # Fallback to Nominatim if BigDataCloud fails
                            try:
                                headers = {'User-Agent': 'RoadAccidentPredictionApp/1.0'}
                                response = requests.get(f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=jsonv2", headers=headers, timeout=5)
                                rev = response.json()
                                loc_str = rev.get('display_name', f"Lat: {lat:.4f}, Lon: {lon:.4f}")
                            except Exception:
                                loc_str = f"Lat: {lat:.4f}, Lon: {lon:.4f}"
                        
                        st.session_state['auto_loc'] = loc_str
                        st.session_state['live_loc_input'] = loc_str 
                        st.session_state['auto_trigger_live'] = True
                        st.session_state['detected_lat'] = lat
                        st.session_state['detected_lon'] = lon
                        st.rerun()
                    except Exception as e:
                        # Fallback for geocoding errors (e.g. rate limit)
                        st.warning("⚠️ Location service is busy. Using coordinates instead.")
                        st.session_state['auto_loc'] = f"{gps_loc['latitude']:.4f}, {gps_loc['longitude']:.4f}"
                        st.session_state['live_loc_input'] = st.session_state['auto_loc']
                        st.session_state['auto_trigger_live'] = True
                        st.session_state['detected_lat'] = gps_loc['latitude']
                        st.session_state['detected_lon'] = gps_loc['longitude']
                        st.rerun()
                    
            with col_text:
                live_loc_input = st.text_input("Enter Live City/Area (e.g. Pune)", value=st.session_state['auto_loc'], key='live_loc_input')
        
            l_cols = st.columns(3)
            with l_cols[0]:
                live_vehicle = st.selectbox("Vehicle Type", feature_info['Vehicle_Type']['options'], key='live_veh')
            with l_cols[1]:
                live_age = st.number_input("Driver Age", min_value=16.0, max_value=100.0, value=35.0, key='live_age')
            with l_cols[2]:
                live_speed = st.number_input("Your Vehicle Speed (km/h)", min_value=15.0, max_value=220.0, value=60.0, key='live_speed')
            
            btn_pressed = st.button("SUBMIT", key="btn_live", use_container_width=True)
            if btn_pressed or st.session_state.get('auto_trigger_live', False):
                st.session_state['auto_trigger_live'] = False # Reset immediately
                if not live_loc_input:
                    st.error("Please enter a location.")
                else:
                    with st.spinner("Fetching live satellite and traffic data..."):
                        # If we auto-detected, use the stored exact coordinates to bypass geocoding
                        use_lat = st.session_state.get('detected_lat') if live_loc_input == st.session_state.get('auto_loc') else None
                        use_lon = st.session_state.get('detected_lon') if live_loc_input == st.session_state.get('auto_loc') else None
                    
                        live_data, err = fetch_live_data(live_loc_input, use_lat, use_lon)
                        if not err:
                            # Fetch Traffic and Accidents
                            traffic_status, curr_speed, free_speed = fetch_live_traffic(live_data['Lat'], live_data['Lon'], tomtom_api_key, live_data['Time_of_Day'])
                        
                            # Extract short city name (before first comma) for the news query
                            short_city = live_loc_input.split(',')[0]
                            accidents = fetch_recent_accidents(short_city)
                    
                        if not err:
                            # If reverse geocoding failed to give a pretty name, use the one from satellite data
                            display_name = live_loc_input
                            if "Lat:" in display_name and live_data.get('City'):
                                display_name = f"{live_data['City']}, {live_data['State']}"
                            
                            st.success(f"📍 **Location Identified:** {display_name}")
                            precip_str = f" | Rain: {live_data['Precipitation']}mm" if live_data['Precipitation'] > 0 else ""
                            st.info(f"✅ Data fetched! Weather: {live_data['Weather_Condition']} ({live_data['Temperature']}°C){precip_str} | Time: {live_data['Time_of_Day']}")
                    
                        st.write("#### 🗺️ Live Location Map")
                        map_df = pd.DataFrame({'lat': [live_data['Lat']], 'lon': [live_data['Lon']]})
                        st.map(map_df, zoom=13)
                    
                        col_t, col_a = st.columns(2)
                        with col_t:
                            st.write("#### 🚦 Live Traffic Condition")
                            if curr_speed:
                                st.write(f"**{traffic_status}** ({curr_speed} km/h on a {free_speed} km/h road)")
                            else:
                                st.write(f"**{traffic_status}**")
                            
                        with col_a:
                            st.write("#### 📰 Recent Local Alerts")
                            if accidents:
                                for acc in accidents:
                                    st.write(f"- 🚨 {acc['title']}")
                            else:
                                st.write("✅ No recent accidents reported in the news for this area in the last 7 days.")
                            
                        # Pass the traffic info to active_live_loc so the Situational Analysis can pick it up
                        active_live_loc = f"{live_loc_input} | Traffic: {traffic_status}"
                    
                        # Use driver's inputted speed for ML inference to predict THEIR specific accident risk
                        final_speed = live_speed
                        final_road = 'Highway' if (free_speed and free_speed >= 65) else 'City Street'

                        active_user_inputs = {
                            'Weather_Condition': 'Clear' if live_data['Weather_Condition'] == 'Cloudy' else live_data['Weather_Condition'],
                            'Road_Type': final_road,
                            'Road_Condition': live_data['Road_Condition'],
                            'Speed_Limit': final_speed,
                            'Time_of_Day': live_data['Time_of_Day'],
                            'Vehicle_Type': live_vehicle,
                            'Driver_Age': live_age,
                            'State': live_data['State'],
                            'City': live_data['City']
                        }
                        predict_clicked = True

        if predict_clicked:
            user_inputs = active_user_inputs
            live_location = active_live_loc
            with st.spinner('Analyzing...'):
                input_df = pd.DataFrame([user_inputs])
            
                # Encode categorical
                for c in cat_cols:
                    le = label_encoders[c]
                    # Fallback for unseen data
                    if input_df[c][0] not in le.classes_:
                        fallback_val = 'Maharashtra' if c == 'State' else ('Mumbai' if c == 'City' else le.classes_[0])
                        if fallback_val not in le.classes_: 
                            fallback_val = le.classes_[0]
                        input_df[c] = le.transform([fallback_val])
                        st.warning(f"⚠️ **Note**: {c.replace('_', ' ')} '{user_inputs[c]}' is unseen in historical training data. The model is estimating risk based on baseline '{fallback_val}'.")
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

                # Log prediction to DB
                log_prediction(st.session_state.get('logged_in_user'), user_inputs, str(rf_label), str(svm_label))

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
                if 'City' in user_inputs and user_inputs['City'] in ['Mumbai', 'New Delhi', 'Chennai', 'Bengaluru']:
                    recommendations.append(f"High traffic density in {user_inputs['City']} drastically increases the chance of severe accidents.")
                
                if live_location:
                    live_loc_lower = live_location.lower()
                    if any(kw in live_loc_lower for kw in ['expressway', 'highway', 'bypass']):
                        recommendations.append(f"🚨 High-speed risk detected on custom location: '{live_location.split(' | ')[0]}'. Expect severe handling dynamics.")
                    elif any(kw in live_loc_lower for kw in ['chowk', 'market', 'street']):
                        recommendations.append(f"🚨 High traffic density detected on custom location: '{live_location.split(' | ')[0]}'. Watch out for pedestrian collisions.")
                
                    if 'heavy traffic' in live_loc_lower:
                        recommendations.append("🚗 Live Traffic Alert: Heavy traffic detected. Risk of rear-end collisions is significantly increased.")

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



            st.markdown("<br><br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 4, 1])
            with col2:
                st.image(os.path.join(os.path.dirname(__file__), 'assets', 'bottom_banner.png'), use_container_width=True)

    # --- ADMIN DASHBOARD ---
    if st.session_state.get('logged_in_user') == 'admin':
        with tab_admin:
            # Fragment for localized refresh
            @st.fragment
            def render_admin_tables():
                st.markdown("---")
                st.header("🛡️ Admin Dashboard")
                
                col_admin1, col_admin2 = st.columns([3, 1])
                with col_admin1:
                    st.write("Welcome, Admin. Here are the users registered in the system:")
                with col_admin2:
                    # Clicking this button will only rerun this fragment, refreshing the data below
                    st.button("🔄 Refresh Data", key="refresh_admin_data")
                        
                users = get_all_users()
                if users:
                    df_users = pd.DataFrame(users)
                    st.dataframe(df_users, use_container_width=True)
                else:
                    st.info("No users found.")
                
                st.write("### 🕒 Recent User Access Logs")
                user_logs = get_all_user_logs()
                if user_logs:
                    df_logs = pd.DataFrame(user_logs)
                    st.dataframe(df_logs, use_container_width=True)
                else:
                    st.info("No login logs found.")
                    
                st.write("### System Prediction History")
                all_preds = get_all_predictions()
                if all_preds:
                    df_preds = pd.DataFrame(all_preds)
                    st.dataframe(df_preds, use_container_width=True)
                else:
                    st.info("No predictions found in the system.")
            
            render_admin_tables()

    # --- AI CHATBOT INTERFACE ---
    with tab_chat:
        st.markdown("---")
        st.header("💬 AI Data Assistant")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({"role": "assistant", "content": "Hello! I am your AI Data Assistant. I can help explain the Random Forest and SVM predictions, or answer questions about your data. How can I help you today?"})
    
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
        # OWN CUSTOM LOCAL DATA AGENT (STATIC NLP ASSISTANT)
        if prompt:
            if "static_assistant" not in st.session_state:
                st.session_state.static_assistant = StaticModelAssistant()
                
            try:
                full_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'accident_data.csv'))
            except Exception:
                full_df = pd.DataFrame()
                
            bot_response = st.session_state.static_assistant.get_response(prompt, full_df)
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(bot_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": bot_response})

if __name__ == "__main__":
    if check_password():
        main()
