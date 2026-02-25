import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Output directory
output_dir = os.getcwd()

def generate_telemetry():
    print("Generating Telemetry Data...")
    units = ['Ball Mill 01', 'Slurry Pump A', 'Conveyor CV-201']
    start_date = datetime(2025, 7, 1)
    end_date = datetime(2025, 12, 31)
    delta = timedelta(hours=1)
    
    data = []
    current_time = start_date
    while current_time <= end_date:
        for unit in units:
            # Base values
            temp = 65 + np.random.normal(0, 2)
            vib = 2.5 + np.random.normal(0, 0.5)
            curr = 150 + np.random.normal(0, 5)
            
            # Simulate a failure pattern for Ball Mill 01 in late September
            if unit == 'Ball Mill 01' and datetime(2025, 9, 15) < current_time < datetime(2025, 9, 25):
                # Gradual increase in temp and vib
                days_since_start = (current_time - datetime(2025, 9, 15)).total_seconds() / 86400
                temp += days_since_start * 3
                vib += days_since_start * 0.8
            
            data.append([current_time, unit, round(temp, 2), round(vib, 2), round(curr, 2)])
        current_time += delta
        
    df = pd.DataFrame(data, columns=['Timestamp', 'UnitName', 'Temperature_C', 'Vibration_mm_s', 'Motor_Current_Amps'])
    df.to_excel(os.path.join(output_dir, 'Telemetry_Sample.xlsx'), index=False)
    print("Saved Telemetry_Sample.xlsx")

def generate_cmms():
    print("Generating CMMS Data...")
    data = [
        ['WO-1001', '2025-07-15 08:00', 'Ball Mill 01', 'Preventive Maintenance', 'Lubrication and Inspection', 450, 'Completed'],
        ['WO-1002', '2025-08-10 14:00', 'Slurry Pump A', 'Corrective Maintenance', 'Seal Replacement', 1200, 'Completed'],
        ['WO-1003', '2025-09-25 10:30', 'Ball Mill 01', 'Emergency Breakdown', 'Bearing Failure - High Temp detected', 8500, 'Completed'],
        ['WO-1004', '2025-10-05 09:00', 'Conveyor CV-201', 'Preventive Maintenance', 'Belt Tensioning', 300, 'Completed'],
        ['WO-1005', '2025-11-12 11:00', 'Slurry Pump A', 'Preventive Maintenance', 'Impeller Check', 500, 'Completed'],
        ['WO-1006', '2025-12-20 07:45', 'Ball Mill 01', 'Preventive Maintenance', 'Full Overhaul', 12000, 'Completed'],
    ]
    df = pd.DataFrame(data, columns=['WorkOrderID', 'StartTime', 'UnitName', 'Type', 'Description', 'Cost_USD', 'Status'])
    df.to_excel(os.path.join(output_dir, 'Maintenance_CMMS_Sample.xlsx'), index=False)
    print("Saved Maintenance_CMMS_Sample.xlsx")

def generate_production():
    print("Generating Production Context Data...")
    start_date = datetime(2025, 7, 1)
    end_date = datetime(2025, 12, 31)
    delta = timedelta(days=1)
    
    data = []
    current_time = start_date
    while current_time <= end_date:
        throughput = 5000 + np.random.normal(0, 200)
        ore_grade = 0.32 + np.random.normal(0, 0.02)
        reagent_idx = 1.5 + np.random.normal(0, 0.1)
        
        data.append([current_time.date(), round(throughput, 1), round(ore_grade, 3), round(reagent_idx, 2)])
        current_time += delta
        
    df = pd.DataFrame(data, columns=['Date', 'Daily_Throughput_Tons', 'Ore_Grade_Pct', 'Reagent_Efficiency_Index'])
    df.to_excel(os.path.join(output_dir, 'Production_Context_Sample.xlsx'), index=False)
    print("Saved Production_Context_Sample.xlsx")

if __name__ == "__main__":
    generate_telemetry()
    generate_cmms()
    generate_production()
