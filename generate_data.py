import pandas as pd
import numpy as np
import random
import os

def generate_mock_data(num_samples=5000, output_file='accident_data.csv'):
    np.random.seed(42)
    random.seed(42)
    
    weather_conditions = ['Clear', 'Rain', 'Snow', 'Fog']
    road_types = ['Highway', 'City Street', 'Rural Road']
    time_of_day = ['Morning', 'Afternoon', 'Evening', 'Night']
    vehicle_types = ['Car', 'Truck', 'Motorcycle', 'Bus']
    
    data = []
    
    for _ in range(num_samples):
        weather = random.choice(weather_conditions)
        road = random.choice(road_types)
        time = random.choice(time_of_day)
        vehicle = random.choice(vehicle_types)
        
        # Base speed limit depending on road type
        if road == 'Highway':
            speed_limit = random.choice([55, 65, 70])
        elif road == 'City Street':
            speed_limit = random.choice([25, 30, 35])
        else:
            speed_limit = random.choice([35, 45, 55])
            
        driver_age = int(np.random.normal(35, 12))
        driver_age = max(16, min(driver_age, 90)) # Bound age between 16 and 90
        
        # Introduce some logic for severity to make the model learnable
        severity_score = 0
        
        if weather in ['Snow', 'Fog']:
            severity_score += 2
        elif weather == 'Rain':
            severity_score += 1
            
        if road == 'Highway':
            severity_score += 2
            
        if vehicle == 'Motorcycle':
            severity_score += 3
            
        if time == 'Night':
            severity_score += 2
            
        if driver_age < 21 or driver_age > 75:
            severity_score += 1
            
        # Add random noise
        severity_score += random.randint(0, 2)
        
        # Categorize severity - adjust thresholds for better balance
        if severity_score <= 2:
            severity = 'Minor'
        elif severity_score <= 4:
            severity = 'Moderate'
        elif severity_score <= 6:
            severity = 'Severe'
        else:
            severity = 'Fatal'
            
        data.append([weather, road, speed_limit, time, vehicle, driver_age, severity])
        
    df = pd.DataFrame(data, columns=['Weather_Condition', 'Road_Type', 'Speed_Limit', 'Time_of_Day', 'Vehicle_Type', 'Driver_Age', 'Accident_Severity'])
    
    # Introduce some missing values to simulate real-world data
    for col in ['Weather_Condition', 'Driver_Age']:
        # Randomly set 2% of the values to NaN
        mask = np.random.rand(num_samples) < 0.02
        df.loc[mask, col] = np.nan

    print(f"Generated {num_samples} samples.")
    print("Severity distribution:")
    print(df['Accident_Severity'].value_counts())
    
    df.to_csv(output_file, index=False)
    print(f"\nSaved data to {output_file}")

if __name__ == "__main__":
    # Get directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'accident_data.csv')
    generate_mock_data(output_file=output_path)
