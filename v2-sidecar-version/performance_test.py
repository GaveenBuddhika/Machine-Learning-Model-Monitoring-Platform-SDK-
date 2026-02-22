import requests
import time
import numpy as np
import pandas as pd

# Configurations
APP_URL = "http://localhost:5000/predict"  
ITERATIONS = 100  

def measure_latency():
    latencies = []
    print(f"Starting Performance Test ({ITERATIONS} iterations)...")

    for i in range(ITERATIONS):
        payload = {
            "income": 5000,
            "loan_amount": 20000,
            "credit_score": 1
        }
        
        start_time = time.perf_counter()
        response = requests.post(APP_URL, json=payload)
        end_time = time.perf_counter()
        
        # Convert to milliseconds
        latency = (end_time - start_time) * 1000
        latencies.append(latency)
        
        if i % 20 == 0:
            print(f"Completed {i} iterations...")

    return latencies

if __name__ == "__main__":
    results = measure_latency()
    
    # Statistical Analysis
    p50 = np.percentile(results, 50)
    p95 = np.percentile(results, 95)
    p99 = np.percentile(results, 99)
    avg_latency = np.mean(results)

    print("\n" + "="*30)
    print("LATENCY PERFORMANCE RESULTS")
    print("="*30)
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"P50 (Median)   : {p50:.2f} ms")
    print(f"P95 (Tail)     : {p95:.2f} ms")
    print(f"P99 (Worst)    : {p99:.2f} ms")
    print("="*30)