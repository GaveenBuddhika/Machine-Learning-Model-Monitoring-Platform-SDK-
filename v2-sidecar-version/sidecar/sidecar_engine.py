from flask import Flask, request, jsonify
import pandas as pd
from prometheus_client import start_http_server, Gauge, Counter, make_wsgi_app
from scipy.stats import ks_2samp
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import os
import psutil
import threading
import time

app = Flask(__name__)

# Metrics definition
DRIFT_SCORE = Gauge('model_drift_score', 'Data drift score using KS Test')
F1_SCORE = Gauge('model_f1_score', 'Real-time model F1-Score')
PRECISION = Gauge('model_precision_score', 'Real-time Precision') 
RECALL = Gauge('model_recall_score', 'Real-time Recall')
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions tracked')

# System metrics
CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('system_memory_usage_bytes', 'Memory usage in bytes')
MEMORY_PERCENT = Gauge('system_memory_usage_percent', 'Memory usage percentage')
DISK_USAGE = Gauge('system_disk_usage_percent', 'Disk usage percentage')

def collect_system_metrics():
    """Background thread to collect system metrics every 5 seconds"""
    while True:
        CPU_USAGE.set(psutil.cpu_percent(interval=1))
        mem = psutil.virtual_memory()
        MEMORY_USAGE.set(mem.used)
        MEMORY_PERCENT.set(mem.percent)
        disk = psutil.disk_usage('/')
        DISK_USAGE.set(disk.percent)
        time.sleep(5)

# Start background thread for system metrics
metrics_thread = threading.Thread(target=collect_system_metrics, daemon=True)
metrics_thread.start()

# Load baseline from shared volume
BASELINE_PATH = os.getenv('BASELINE_PATH', 'data/baseline_data.csv')
baseline_df = pd.read_csv(BASELINE_PATH)

# Global variables for F1 calculation
tp, fp, tn, fn = 0, 0, 0, 0

@app.route('/track', methods=['POST'])
def track():
    global tp, fp, tn, fn
    data = request.json
    
    # 1. Generic Drift Detection (Compares index 1 of input vs baseline)
    val = data['features'][1] # Assuming loan_amount or primary feature is at index 1
    stat, _ = ks_2samp(baseline_df.iloc[:, 1], [val])
    DRIFT_SCORE.set(stat)

    # 2. F1-Score Logic
    pred = data['prediction']
    actual = data.get('actual')
    if actual is not None:
        if pred == 1 and actual == 1: tp += 1
        elif pred == 1 and actual == 0: fp += 1
        elif pred == 0 and actual == 0: tn += 1
        elif pred == 0 and actual == 1: fn += 1
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        F1_SCORE.set(f1)
        PRECISION.set(prec)
        RECALL.set(rec)

    PREDICTION_COUNT.inc()
    return jsonify({"status": "tracked"}), 200

# Expose /metrics for Prometheus
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {'/metrics': make_wsgi_app()})

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8000)