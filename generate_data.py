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
    haryana_districts = ['Ambala', 'Bhiwani', 'Charkhi Dadri', 'Faridabad', 'Fatehabad', 'Gurugram', 'Hisar', 'Jhajjar', 'Jind', 'Kaithal', 'Karnal', 'Kurukshetra', 'Mahendragarh', 'Nuh', 'Palwal', 'Panchkula', 'Panipat', 'Rewari', 'Rohtak', 'Sirsa', 'Sonipat', 'Yamunanagar']
    delhi_districts = ['Central Delhi', 'East Delhi', 'New Delhi', 'North Delhi', 'North East Delhi', 'North West Delhi', 'Shahdara', 'South Delhi', 'South East Delhi', 'South West Delhi', 'West Delhi']
    up_districts = ['Agra', 'Aligarh', 'Prayagraj', 'Ambedkar Nagar', 'Amethi', 'Amroha', 'Auraiya', 'Ayodhya', 'Azamgarh', 'Baghpat', 'Bahraich', 'Ballia', 'Balrampur', 'Banda', 'Barabanki', 'Bareilly', 'Basti', 'Bhadohi', 'Bijnor', 'Budaun', 'Bulandshahr', 'Chandauli', 'Chitrakoot', 'Deoria', 'Etah', 'Etawah', 'Farrukhabad', 'Fatehpur', 'Firozabad', 'Gautam Buddha Nagar', 'Ghaziabad', 'Ghazipur', 'Gonda', 'Gorakhpur', 'Hamirpur', 'Hapur', 'Hardoi', 'Hathras', 'Jalaun', 'Jaunpur', 'Jhansi', 'Kannauj', 'Kanpur Dehat', 'Kanpur Nagar', 'Kasganj', 'Kaushambi', 'Lakhimpur Kheri', 'Kushinagar', 'Lalitpur', 'Lucknow', 'Maharajganj', 'Mahoba', 'Mainpuri', 'Mathura', 'Mau', 'Meerut', 'Mirzapur', 'Moradabad', 'Muzaffarnagar', 'Pilibhit', 'Pratapgarh', 'Raebareli', 'Rampur', 'Saharanpur', 'Sambhal', 'Sant Kabir Nagar', 'Shahjahanpur', 'Shamli', 'Shravasti', 'Siddharthnagar', 'Sitapur', 'Sonbhadra', 'Sultanpur', 'Unnao', 'Varanasi']
    
    state_cities = {
        'Delhi': delhi_districts,
        'Haryana': haryana_districts,
        'Maharashtra': ['Mumbai', 'Pune', 'Nagpur', 'Nashik'],
        'Karnataka': ['Bengaluru', 'Mysuru', 'Mangaluru', 'Hubli'],
        'Tamil Nadu': ['Chennai', 'Coimbatore', 'Madurai', 'Salem'],
        'Uttar Pradesh': up_districts
    }
    
    data = []
    
    for _ in range(num_samples):
        weather = random.choice(weather_conditions)
        road = random.choice(road_types)
        road_cond = random.choice(['Normal', 'Wet', 'Potholes', 'Hill Area'])
        time = random.choice(time_of_day)
        vehicle = random.choice(vehicle_types)
        state = random.choice(list(state_cities.keys()))
        city = random.choice(state_cities[state])
        
        # Base speed limit depending on road type
        if road == 'Highway':
            speed_limit = random.choice([55, 65, 70, 80, 100, 120, 150, 200, 220])
        elif road == 'City Street':
            speed_limit = random.choice([25, 30, 35, 45, 60, 80])
        else:
            speed_limit = random.choice([35, 45, 55, 70, 90, 120])
            
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
            
        if speed_limit >= 120:
            severity_score += 4
        elif speed_limit >= 80:
            severity_score += 2
        elif speed_limit >= 65:
            severity_score += 1
            
        if road_cond == 'Hill Area':
            severity_score += 3
        elif road_cond == 'Potholes' or road_cond == 'Wet':
            severity_score += 1
            
        if city in ['Mumbai', 'New Delhi', 'Chennai', 'Bengaluru']:
            severity_score += 2
        elif city in ['Pune', 'Lucknow', 'Kanpur']:
            severity_score += 1
        elif city in ['Mysuru', 'Salem', 'Dwarka']:
            severity_score -= 1
            
        # Add very minimal random noise (0 or 1) so models can achieve higher accuracy
        severity_score += random.randint(0, 1)
        
        # Categorize severity - adjust thresholds for better balance
        if severity_score <= 2:
            severity = 'Minor'
        elif severity_score <= 4:
            severity = 'Moderate'
        elif severity_score <= 6:
            severity = 'Severe'
        else:
            severity = 'Fatal'
            
        data.append([weather, road, road_cond, speed_limit, time, vehicle, driver_age, state, city, severity])
        
    df = pd.DataFrame(data, columns=['Weather_Condition', 'Road_Type', 'Road_Condition', 'Speed_Limit', 'Time_of_Day', 'Vehicle_Type', 'Driver_Age', 'State', 'City', 'Accident_Severity'])
    
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
