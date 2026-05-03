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
from database import init_db, add_user, verify_user, reset_password, log_prediction, get_predictions, get_all_users, get_all_predictions
from notifications import notify_admin_of_new_user, notify_user_of_registration
from static_assistant import StaticModelAssistant

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
            geo_resp = requests.get(geocode_url).json()
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
            geo_resp = requests.get(geocode_url).json()
            if 'results' in geo_resp and geo_resp['results']:
                state = geo_resp['results'][0].get('admin1', 'Delhi')
                city = geo_resp['results'][0].get('name', 'New Delhi')
        
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code"
        weather_resp = requests.get(weather_url).json()
        current = weather_resp.get('current', {})
        w_code = current.get('weather_code', 0)
        temp = current.get('temperature_2m', 0)
        
        if w_code in [0, 1, 2, 3]: weather = 'Clear'
        elif w_code in [45, 48]: weather = 'Fog'
        elif w_code in [71, 73, 75, 77, 85, 86]: weather = 'Snow'
        else: weather = 'Rain'
        
        current_hour = datetime.now().hour
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
st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    .stApp { 
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
        font-family: 'Inter', sans-serif;
    }
    
    h1 { 
        font-weight: 800; 
        color: #1a1a1a; 
        text-align: center; 
        margin-bottom: 5px !important; 
        letter-spacing: -1px;
    }
    
    .subtitle { 
        text-align: center; 
        color: #4a4a4a; 
        font-size: 1.2em; 
        margin-bottom: 40px; 
        font-weight: 400;
    }
    
    .stMarkdown p, .stMarkdown li {
        color: #2d3436;
        font-size: 1.05em;
    }
    
    .prediction-card { 
        background-color: white; 
        padding: 35px; 
        border-radius: 20px; 
        box-shadow: 0 15px 35px rgba(0,0,0,0.08); 
        text-align: center; 
        margin-top: 30px; 
        border: 1px solid #f1f3f5;
    }
    
    .pred-value { 
        font-size: 2.5em; 
        font-weight: 800; 
        color: #0984e3; 
    }
    
    /* Make labels more visible */
    .stSelectbox label, .stTextInput label, .stNumberInput label {
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 1.15em !important;
    }

    /* Make the input boxes themselves extremely visible with thick borders */
    .stTextInput input, .stSelectbox [data-baseweb="select"], .stNumberInput input, .stTextArea textarea {
        border: 3px solid #2d3436 !important;
        border-radius: 12px !important;
        padding: 12px !important;
        background-color: #ffffff !important;
        color: #000000 !important;
        font-size: 1.1em !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }

    /* Highlight box when user clicks on it */
    .stTextInput input:focus, .stSelectbox [data-baseweb="select"]:focus, .stNumberInput input:focus {
        border-color: #0984e3 !important;
        box-shadow: 0 0 0 4px rgba(9, 132, 227, 0.3) !important;
    }

    /* Make placeholders clearly visible but hide them on focus */
    ::placeholder {
        color: #adb5bd !important;
        opacity: 1 !important;
        transition: color 0.2s ease, opacity 0.2s ease;
    }
    
    /* Strongest possible way to hide placeholder on click/focus */
    input:focus::placeholder, textarea:focus::placeholder {
        color: transparent !important;
        opacity: 0 !important;
    }
    
    /* Browser specific overrides for focus */
    input:focus::-webkit-input-placeholder { color: transparent !important; }
    input:focus:-moz-placeholder { color: transparent !important; }
    input:focus::-moz-placeholder { color: transparent !important; }
    input:focus:-ms-input-placeholder { color: transparent !important; }

    /* Improve Form appearance and hide 'Press Enter to submit' hint */
    [data-testid="stForm"] {
        background-color: #ffffff !important;
        padding: 40px !important;
        border-radius: 25px !important;
        border: 3px solid #dfe6e9 !important;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1) !important;
    }
    
    /* Hide the 'Press Enter to submit' hint (Ultimate Global Fix) */
    [data-testid="stForm"] small, 
    .stFormSubmitButton small, 
    div[data-testid="stFormSubmitButton"] p,
    div[data-testid="stForm"] [data-testid="stMarkdownContainer"] p:only-child {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
        opacity: 0 !important;
        font-size: 0px !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Mobile Responsiveness for inputs and placeholders */
    @media (max-width: 640px) {
        .stTextInput input, .stSelectbox [data-baseweb="select"], .stNumberInput input {
            font-size: 14px !important;
            padding: 8px !important;
        }
        ::placeholder {
            font-size: 13px !important;
        }
        h1 {
            font-size: 2em !important;
        }
        .subtitle {
            font-size: 1em !important;
        }
    }
</style>""", unsafe_allow_html=True)



# --- USER AUTHENTICATION & DATABASE ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered(username_val, password_val):
        # Normalize username to lowercase and trim spaces
        username_val = username_val.strip().lower() if username_val else ""
        if verify_user(username_val, password_val):
            st.session_state["password_correct"] = True
            st.session_state["logged_in_user"] = username_val
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state or not st.session_state["password_correct"]:
        # Dynamic Greeting based on India Time (Kolkata)
        ist = pytz.timezone('Asia/Kolkata')
        india_time = datetime.datetime.now(ist)
        hour = india_time.hour
        
        if hour < 12: greeting = "Good Morning ☀️"
        elif hour < 18: greeting = "Good Afternoon 🌤️"
        else: greeting = "Good Evening 🌙"
        
        st.markdown(f"""
        <div style='text-align: center; margin-top: 30px; animation: fadeIn 1s ease-in;'>
            <h1 style='font-size: 3.5em; margin-bottom: 0;'>{greeting}</h1>
            <p style='font-size: 1.3em; color: #636e72;'>Safety first. Predict risks, save lives.</p>
        </div>
        
        <style>
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .stTabs [data-baseweb="tab-list"] {{
                gap: 10px;
                justify-content: center;
            }}
            .stTabs [data-baseweb="tab"] {{
                background-color: #f1f3f5;
                border-radius: 10px 10px 0 0;
                padding: 10px 20px;
                color: #4b6584;
                transition: all 0.3s;
            }}
            .stTabs [aria-selected="true"] {{
                background-color: #0984e3 !important;
                color: white !important;
            }}
        </style>
        """, unsafe_allow_html=True)
        
        # Add a Lottie Animation for interactivity
        lottie_traffic = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_5njpXm.json")
        if lottie_traffic:
            st_lottie(lottie_traffic, height=200, key="traffic_lottie")
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            tab1, tab2 = st.tabs(["🔑 Sign In", "📝 Create Account"])
            
            with tab1:
                st.markdown("<div style='padding: 10px;'>", unsafe_allow_html=True)
                with st.form("login_form"):
                    st.markdown("### 🔒 Secure Login")
                    st.text_input("👤 Enter your Login ID", key="username", placeholder="Your Login ID")
                    st.text_input("🔑 Enter your Password", type="password", key="password", placeholder="Your Password")
                    submit_login = st.form_submit_button("Access Dashboard 🚀", use_container_width=True)
                
                if submit_login:
                    password_entered(st.session_state.username, st.session_state.password)
                    st.rerun()
                
                if "password_correct" in st.session_state and not st.session_state["password_correct"]:
                    st.error("😕 Username or password incorrect")
                    st.info("💡 **Tip:** If you just registered on your local computer, you must register again on this Cloud URL, as they use separate databases.")
                    
                with st.expander("❓ Forgot Password?"):
                    if st.session_state.get('password_reset_success', False):
                        st.success("✅ Password changed successfully!")
                        st.info("You can now sign in with your new password.")
                        if st.button("↩️ Back to Sign In", key="back_to_signin"):
                            st.session_state.password_reset_success = False
                            st.rerun()
                    else:
                        st.write("Reset your password by verifying your registered email.")
                        with st.form("reset_form"):
                            reset_user = st.text_input("👤 Username", key="reset_username", placeholder="Your Login ID")
                            reset_email = st.text_input("📧 Registered Email", key="reset_email", placeholder="email@example.com")
                            reset_new_pass = st.text_input("🆕 New Password", type="password", key="reset_new_password", placeholder="Choose a strong password")
                            submit_reset = st.form_submit_button("Reset My Password", use_container_width=True)
                        
                        if submit_reset:
                            if reset_user == "" or reset_email == "" or reset_new_pass == "":
                                st.error("⚠️ All fields must be filled.")
                            else:
                                success, message = reset_password(reset_user, reset_email, reset_new_pass)
                                if success:
                                    st.session_state.password_reset_success = True
                                    st.rerun()
                                else:
                                    st.error(f"⚠️ {message}")
                st.markdown("</div>", unsafe_allow_html=True)
                                
            with tab2:
                st.markdown("<div style='padding: 10px;'>", unsafe_allow_html=True)
                if st.session_state.get('registration_success', False):
                    st.success(f"🎉 Account created successfully!")
                    st.info(f"**IMPORTANT: Your generated Login ID is:** `{st.session_state.new_login_id}`\n\nPlease switch to the Sign In tab and use this Login ID to sign in.")
                    if st.button("⬅️ Register Another Account"):
                        st.session_state.registration_success = False
                        st.rerun()
                else:
                    st.markdown("<h3 style='color: #000000; font-weight: 800;'>📝 New Registration</h3>", unsafe_allow_html=True)
                    with st.form("registration_form"):
                        r_col1, r_col2 = st.columns(2)
                        with r_col1:
                            first_name = st.text_input("👤 Enter First Name", placeholder="e.g. Himanshu")
                        with r_col2:
                            last_name = st.text_input("👤 Enter Last Name", placeholder="e.g. Prajapati")
                        
                        new_email = st.text_input("📧 Enter Email Address", placeholder="your@email.com")
                        new_contact = st.text_input("📱 Enter Contact Number", placeholder="+91 XXXXX XXXXX")
                        new_pass = st.text_input("🔐 Create a Strong Password", type="password", placeholder="Choose a password")
                        
                        submitted = st.form_submit_button("CREATE MY ACCOUNT ✅", use_container_width=True)
                    
                    if submitted:
                        if not first_name or not last_name or not new_email or not new_pass:
                            st.error("⚠️ All required fields must be filled.")
                        else:
                            base_username = f"{first_name.strip().lower()}{last_name.strip().lower()}"
                            import random
                            new_user = base_username
                            
                            # Try to generate a unique username
                            for _ in range(10):
                                success, message = add_user(
                                    username=new_user, 
                                    password=new_pass, 
                                    email=new_email,
                                    full_name=f"{first_name.strip()} {last_name.strip()}",
                                    contact_number=new_contact
                                )
                                if success:
                                    notify_admin_of_new_user(new_user, new_email)
                                    notify_user_of_registration(new_email, new_contact, first_name.strip(), new_user)
                                    
                                    st.session_state.registration_success = True
                                    st.session_state.new_login_id = new_user
                                    st.rerun()
                                else:
                                    new_user = f"{base_username}{random.randint(10, 9999)}"
                            else:
                                st.error("⚠️ Could not generate a unique username. Please try again.")
                st.markdown("</div>", unsafe_allow_html=True)
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
        for key in st.session_state.keys():
            del st.session_state[key]
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
            if st.button("Predict Outcome 🎯", key="btn_manual", use_container_width=True):
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
                    
                        # Reverse geocode with Nominatim to get exact address
                        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                        rev = requests.get(f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=jsonv2", headers=headers, timeout=5).json()
                    
                        loc_str = rev.get('display_name') or "Unknown Location"
                        st.session_state['auto_loc'] = loc_str
                        st.session_state['live_loc_input'] = loc_str # Force widget update
                        st.session_state['auto_trigger_live'] = True
                        # Pass the exact coordinates forward to bypass geocoding issues with long names
                        st.session_state['detected_lat'] = lat
                        st.session_state['detected_lon'] = lon
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not reverse geocode: {e}")
                    
            with col_text:
                live_loc_input = st.text_input("Enter Live City/Area (e.g. Pune)", value=st.session_state['auto_loc'], key='live_loc_input')
        
            l_cols = st.columns(3)
            with l_cols[0]:
                live_vehicle = st.selectbox("Vehicle Type", feature_info['Vehicle_Type']['options'], key='live_veh')
            with l_cols[1]:
                live_age = st.number_input("Driver Age", min_value=16.0, max_value=100.0, value=35.0, key='live_age')
            with l_cols[2]:
                live_speed = st.number_input("Your Vehicle Speed (km/h)", min_value=15.0, max_value=220.0, value=60.0, key='live_speed')
            
            btn_pressed = st.button("Fetch Live Data & Predict 🚀", key="btn_live", use_container_width=True)
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
                    
                    if err:
                        st.error(f"Error fetching data: {err}")
                    else:
                        st.success(f"✅ Data fetched! Current Weather: {live_data['Weather_Condition']} ({live_data['Temperature']}°C) | Time: {live_data['Time_of_Day']}")
                    
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
                            'Weather_Condition': live_data['Weather_Condition'],
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
