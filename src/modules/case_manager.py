import json
import os
import csv
from datetime import datetime

class CaseManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.json_file = os.path.join(self.data_dir, "cases.json")
        self.csv_file = os.path.join(self.data_dir, "cases.csv")
        
        if not os.path.exists(self.json_file):
            with open(self.json_file, 'w') as f:
                json.dump([], f)
                
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Case ID", "Timestamp", "Filename", "Prediction", "Confidence", "Risk Type", "Threat Level", "Trust Score", "Action"])

    def save_case(self, case_id, filename, prediction, confidence, risk_type, threat_level, trust_score, action):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        case_data = {
            "Case ID": case_id,
            "Timestamp": timestamp,
            "Filename": filename,
            "Prediction": prediction,
            "Confidence": round(float(confidence), 2),
            "Risk Type": risk_type,
            "Threat Level": threat_level,
            "Trust Score": round(float(trust_score), 2),
            "Action": action
        }
        
        # Save to JSON
        with open(self.json_file, 'r') as f:
             try:
                 cases = json.load(f)
             except json.JSONDecodeError:
                 cases = []
                 
        cases.append(case_data)
        with open(self.json_file, 'w') as f:
            json.dump(cases, f, indent=4)
            
        # Save to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([case_id, timestamp, filename, prediction, confidence, risk_type, threat_level, trust_score, action])
            
        return case_data
