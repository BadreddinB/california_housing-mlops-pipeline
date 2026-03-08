import os
import subprocess

DRIFT_REPORT_PATH = "monitoring/drift_report.html"

def drift_detected():
    if not os.path.exists(DRIFT_REPORT_PATH):
        return False
    
    with open(DRIFT_REPORT_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    
    return "drift detected" in html.lower()

def retrain_model():
    print("Drift detected. Retraining model...")
    subprocess.run(["python", "src/train.py"])
    print("Retraining complete.")

if __name__ == "__main__":
    if drift_detected():
        retrain_model()
    else:
        print("No drift detected. No retraining needed.")