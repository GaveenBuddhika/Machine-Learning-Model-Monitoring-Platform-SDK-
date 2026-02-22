import requests
import time

# Configurations
URL = "http://localhost:5000/predict"
NORMAL_DATA = {
    "income": 5000,
    "loan_amount": 20000,
    "credit_score": 1
}

def stabilize_system(iterations=20):
    print(f"🔄 Stabilizing system with {iterations} normal requests...")
    
    for i in range(1, iterations + 1):
        try:
            response = requests.post(URL, json=NORMAL_DATA)
            if response.status_code == 200:
                print(f"[{i}/{iterations}] Request sent successfully.")
            else:
                print(f"[{i}/{iterations}] Failed with status: {response.status_code}")
        except Exception as e:
            print(f"Error: {e}")
        
        # තත්පරයක විවේකයක් ගනිමු (Grafana එකට දත්ත update වෙන්න වෙලාව දෙන්න)
        time.sleep(1)

    print("\n✅ System Stabilized! Now check Grafana for a low K-S Score.")

if __name__ == "__main__":
    stabilize_system()