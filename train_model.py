import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'accident_data.csv')
    
    if not os.path.exists(data_path):
        print("Data file not found. Please run generate_data.py first.")
        return

    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # 1. Handle missing values
    # For categorical columns, fill with the most frequent value
    cat_cols = ['Weather_Condition']
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    # For numerical columns, fill with the median
    num_cols = ['Driver_Age']
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # 2. Encode categorical features
    label_encoders = {}
    features_to_encode = ['Weather_Condition', 'Road_Type', 'Road_Condition', 'Time_of_Day', 'Vehicle_Type', 'State', 'City']
    
    for col in features_to_encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    # Save label encoders for later use during inference
    joblib.dump(label_encoders, os.path.join(current_dir, 'label_encoders.pkl'))

    # Encode target variable
    target_le = LabelEncoder()
    df['Accident_Severity'] = target_le.fit_transform(df['Accident_Severity'])
    joblib.dump(target_le, os.path.join(current_dir, 'target_encoder.pkl'))

    # 3. Define features (X) and target (y)
    X = df.drop('Accident_Severity', axis=1)
    y = df['Accident_Severity']

    # 4. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(current_dir, 'scaler.pkl'))

    # 6. Train the models
    print("Training Random Forest Model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    print("Training SVM Model...")
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)

    # 7. Evaluate the models
    target_names = target_le.inverse_transform(np.unique(y_test))
    
    print("\n--- Random Forest Evaluation ---")
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"Accuracy: {rf_accuracy:.4f}")
    print(classification_report(y_test, rf_pred, target_names=target_names))
    
    print("\n--- SVM Evaluation ---")
    svm_pred = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    print(f"Accuracy: {svm_accuracy:.4f}")
    print(classification_report(y_test, svm_pred, target_names=target_names))

    # 8. Save the models
    rf_path = os.path.join(current_dir, 'rf_model.pkl')
    joblib.dump(rf_model, rf_path)
    
    svm_path = os.path.join(current_dir, 'svm_model.pkl')
    joblib.dump(svm_model, svm_path)
    print(f"\nModels saved to {current_dir}")

if __name__ == "__main__":
    main()
