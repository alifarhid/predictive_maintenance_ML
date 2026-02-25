import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

# Paths
data_dir = os.getcwd()
tel_file = os.path.join(data_dir, 'Telemetry_Sample.xlsx')
maint_file = os.path.join(data_dir, 'Maintenance_CMMS_Sample.xlsx')
prod_file = os.path.join(data_dir, 'Production_Context_Sample.xlsx')

def build_model():
    print("Step 1: Loading Datasets...")
    df_tel = pd.read_excel(tel_file)
    df_maint = pd.read_excel(maint_file)
    df_prod = pd.read_excel(prod_file)
    
    # Pre-process Timestamps
    df_tel['Timestamp'] = pd.to_datetime(df_tel['Timestamp'])
    df_maint['StartTime'] = pd.to_datetime(df_maint['StartTime'])
    df_prod['Date'] = pd.to_datetime(df_prod['Date'])
    
    print("Step 2: Feature Engineering (Rolling Windows)...")
    # Hourly telemetry features
    df_tel = df_tel.sort_values(['UnitName', 'Timestamp'])
    for col in ['Temperature_C', 'Vibration_mm_s', 'Motor_Current_Amps']:
        # 24-hour rolling mean
        df_tel[f'{col}_Mean_24h'] = df_tel.groupby('UnitName')[col].transform(lambda x: x.rolling(window=24, min_periods=1).mean())
        # 24-hour rolling std (instability indicator)
        df_tel[f'{col}_Std_24h'] = df_tel.groupby('UnitName')[col].transform(lambda x: x.rolling(window=24, min_periods=1).std())
    
    print("Step 3: Merging Production Context...")
    df_tel['Date'] = df_tel['Timestamp'].dt.normalize()
    df_final = df_tel.merge(df_prod, on='Date', how='left')
    
    print("Step 4: Labeling (Target Variable)...")
    # Objective: Predict if an 'Emergency Breakdown' will occur in the next 24 hours
    breakdowns = df_maint[df_maint['Type'] == 'Emergency Breakdown'][['StartTime', 'UnitName']]
    
    df_final['Fail_Next_24h'] = 0
    for idx, row in breakdowns.iterrows():
        # Mark all records for that unit in the 24 hours BEFORE the breakdown as '1'
        window_start = row['StartTime'] - pd.Timedelta(hours=24)
        window_end = row['StartTime']
        
        mask = (df_final['UnitName'] == row['UnitName']) & \
               (df_final['Timestamp'] >= window_start) & \
               (df_final['Timestamp'] < window_end)
        df_final.loc[mask, 'Fail_Next_24h'] = 1
        
    print(f"Target Distribution:\n{df_final['Fail_Next_24h'].value_counts()}")
    
    print("Step 5: Model Training...")
    # Features
    features = [
        'Temperature_C_Mean_24h', 'Temperature_C_Std_24h',
        'Vibration_mm_s_Mean_24h', 'Vibration_mm_s_Std_24h',
        'Motor_Current_Amps_Mean_24h', 'Motor_Current_Amps_Std_24h',
        'Daily_Throughput_Tons', 'Ore_Grade_Pct'
    ]
    
    # Drop rows with NaNs (from rolling window start)
    df_ml = df_final.dropna(subset=features)
    
    X = df_ml[features]
    y = df_ml['Fail_Next_24h']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    print("\n--- Model Performance Report ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    
    print("\nFeature Importance:")
    importance = pd.DataFrame({'Feature': features, 'Importance': clf.feature_importances_}).sort_values(by='Importance', ascending=False)
    print(importance)

if __name__ == "__main__":
    build_model()
