# 🚗 Road Accident Prediction Platform

A comprehensive, real-time, Machine Learning-powered web application built with Streamlit. This platform leverages Random Forest and Support Vector Machine (SVM) algorithms to predict the severity of road accidents based on environmental, vehicle, and driver conditions. 

It also includes real-time situational awareness via live weather tracking, TomTom Traffic APIs, Google News alerts, and a dynamic Gemini AI Chatbot assistant.

---

## 🌟 Key Features

1. **AutoML & Dynamic Prediction**: 
   - Upload any custom CSV dataset, pick a target column, and the system dynamically trains new RF and SVM models in the background.
   - Falls back to robust default models trained on synthetic Indian road accident data if no custom data is provided.
2. **Live Telemetry & Situational Risk Analysis**:
   - Auto-fetches live user location via GPS.
   - Pulls real-time weather conditions via Open-Meteo API.
   - Fetches live traffic speeds using the TomTom Traffic API.
   - Scrapes recent local road accident news alerts to warn users of immediate hazards.
3. **Secure User Authentication System**:
   - Secure local SQLite database (`app_data.db`).
   - Secure hashed password storage.
   - "Forgot Password" feature with email verification constraints.
   - Admin Dashboard (`admin` user) to view all users and global prediction history.
4. **AI Data Assistant**:
   - Integrated with Google Gemini 1.5 Flash.
   - Users can chat with the AI to get explanations of the predictions and dynamic insights into the datasets.

---

## 📁 Project Structure

```text
road_accident_prediction/
│
├── app.py                  # Main Streamlit application and UI
├── database.py             # SQLite Database handler, Auth, & Logging
├── generate_data.py        # Script to generate synthetic accident_data.csv
├── train_model.py          # Script to preprocess data and train default ML models
├── test_pred.py            # CLI script to test extreme severity inputs
├── requirements.txt        # Project dependencies
│
├── .env                    # Environment variables (API Keys, DB Configs)
├── .gitignore              # Git ignore configurations
│
└── Models & Data (Auto-Generated):
    ├── accident_data.csv   # The default dataset
    ├── app_data.db         # The local SQLite database
    ├── rf_model.pkl        # Saved Random Forest model
    ├── svm_model.pkl       # Saved Support Vector Machine model
    ├── scaler.pkl          # Standard Scaler
    └── label_encoders.pkl  # Encoders for categorical features
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.9+
- Git

### 2. Clone and Install
```bash
# Navigate to project directory
cd road_accident_prediction

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Mac/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 3. Generate Data and Train Models
If you are running the project for the very first time, you need to build the baseline models:
```bash
# 1. Generate the synthetic dataset
python generate_data.py

# 2. Train and save the Random Forest and SVM models
python train_model.py
```

### 4. Environment Variables
Create a `.env` file in the root directory (this is automatically ignored by Git) to activate live APIs:
```ini
GEMINI_API_KEY=your_google_gemini_api_key_here
```
*(You can also configure TomTom API keys directly inside the application's sidebar)*

---

## 🖥️ Usage

To start the Streamlit web server, run:
```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

### User Flow
1. **Register/Login**: Create an account or log in. You can also securely reset your password using your registered email.
2. **Dashboard**: Navigate the primary dashboard.
3. **Manual Prediction**: Enter specific parameters (City, Vehicle Type, Age, Speed Limit) to test hypothetical scenarios.
4. **Live Prediction**: Click the GPS button to fetch live environmental data and run a real-time risk assessment.
5. **AI Chat**: Open the sidebar, enter your Gemini API key, and chat with the dataset!

---

## 🤖 Algorithms Used

- **Random Forest Classifier**: Utilized for its robustness against overfitting and ability to handle both categorical and numerical data efficiently. Used as the primary high-accuracy predictor.
- **Support Vector Machine (RBF Kernel)**: Used as a secondary comparative model to establish decision boundaries. Due to SVM's $O(n^3)$ time complexity, the dynamic pipeline subsamples large datasets (n > 2000) to maintain rapid UI responsiveness.

---

## ☁️ Deployment

This project is optimized for deployment on **Streamlit Community Cloud**:
1. Commit and push this directory to a public/private GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account and point it to `app.py`.
4. Ensure you add your `GEMINI_API_KEY` to the **Secrets** section in the Streamlit Cloud advanced settings!
