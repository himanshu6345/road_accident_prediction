import joblib
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

rf_model = joblib.load(os.path.join(current_dir, 'rf_model.pkl'))
svm_model = joblib.load(os.path.join(current_dir, 'svm_model.pkl'))
scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))
label_encoders = joblib.load(os.path.join(current_dir, 'label_encoders.pkl'))
target_encoder = joblib.load(os.path.join(current_dir, 'target_encoder.pkl'))

def predict(weather, road_type, road_cond, speed_limit, time_of_day, vehicle_type, driver_age, state, city):
    input_data = pd.DataFrame({
        'Weather_Condition': [weather],
        'Road_Type': [road_type],
        'Road_Condition': [road_cond],
        'Speed_Limit': [speed_limit],
        'Time_of_Day': [time_of_day],
        'Vehicle_Type': [vehicle_type],
        'Driver_Age': [driver_age],
        'State': [state],
        'City': [city]
    })
    
    for col in ['Weather_Condition', 'Road_Type', 'Road_Condition', 'Time_of_Day', 'Vehicle_Type', 'State', 'City']:
        le = label_encoders[col]
        input_data[col] = le.transform(input_data[col])
        
    # Reorder columns to match the training data
    input_ordered = input_data[['Weather_Condition', 'Road_Type', 'Road_Condition', 'Speed_Limit', 'Time_of_Day', 'Vehicle_Type', 'Driver_Age', 'State', 'City']]
    input_scaled = scaler.transform(input_ordered)
    
    rf_pred = target_encoder.inverse_transform(rf_model.predict(input_scaled))[0]
    svm_pred = target_encoder.inverse_transform(svm_model.predict(input_scaled))[0]
    
    print(f"Inputs: {weather}, {road_type}, {road_cond}, {speed_limit}, {time_of_day}, {vehicle_type}, {driver_age}, {state}, {city}")
    print(f"RF Prediction: {rf_pred}")
    print(f"SVM Prediction: {svm_pred}\n")

print("Testing extremes:")
# Extreme low severity
predict('Clear', 'City Street', 'Normal', 25, 'Morning', 'Car', 40, 'Delhi', 'New Delhi')
# Default UI severity
predict('Clear', 'Highway', 'Normal', 45, 'Morning', 'Car', 35, 'Delhi', 'New Delhi')
# Extreme high severity
predict('Snow', 'Highway', 'Hill Area', 70, 'Night', 'Motorcycle', 18, 'Maharashtra', 'Mumbai')
