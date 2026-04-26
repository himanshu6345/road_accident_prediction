import joblib
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

rf_model = joblib.load(os.path.join(current_dir, 'rf_model.pkl'))
svm_model = joblib.load(os.path.join(current_dir, 'svm_model.pkl'))
scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))
label_encoders = joblib.load(os.path.join(current_dir, 'label_encoders.pkl'))
target_encoder = joblib.load(os.path.join(current_dir, 'target_encoder.pkl'))

def predict(weather, road_type, speed_limit, time_of_day, vehicle_type, driver_age):
    input_data = pd.DataFrame({
        'Weather_Condition': [weather],
        'Road_Type': [road_type],
        'Speed_Limit': [speed_limit],
        'Time_of_Day': [time_of_day],
        'Vehicle_Type': [vehicle_type],
        'Driver_Age': [driver_age]
    })
    
    for col in ['Weather_Condition', 'Road_Type', 'Time_of_Day', 'Vehicle_Type']:
        le = label_encoders[col]
        input_data[col] = le.transform(input_data[col])
        
    input_scaled = scaler.transform(input_data)
    
    rf_pred = target_encoder.inverse_transform(rf_model.predict(input_scaled))[0]
    svm_pred = target_encoder.inverse_transform(svm_model.predict(input_scaled))[0]
    
    print(f"Inputs: {weather}, {road_type}, {speed_limit}, {time_of_day}, {vehicle_type}, {driver_age}")
    print(f"RF Prediction: {rf_pred}")
    print(f"SVM Prediction: {svm_pred}\n")

print("Testing extremes:")
# Extreme low severity
predict('Clear', 'City Street', 25, 'Morning', 'Car', 40)
# Default UI severity
predict('Clear', 'Highway', 45, 'Morning', 'Car', 35)
# Extreme high severity
predict('Snow', 'Highway', 70, 'Night', 'Motorcycle', 18)
