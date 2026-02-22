from flask import Flask, render_template, request, jsonify
import joblib
import requests
import os
from prometheus_client import make_wsgi_app, Counter
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('app_requests_total', 'Total requests to the ML app')
PREDICTION_COUNT = Counter('app_predictions_total', 'Total predictions made')

# Load model from shared volume
model = joblib.load('models/loan_model.joblib')

def push_to_sidecar(features, pred, actual):
    payload = {"features": features, "prediction": int(pred), "actual": int(actual) if actual else None}
    try:
        requests.post("http://sidecar-monitor:8000/track", json=payload, timeout=0.5)
    except:
        pass # Sidecar down shouldn't break the app

@app.route('/predict', methods=['POST'])
def predict_api():
    """JSON API endpoint for predictions"""
    REQUEST_COUNT.inc()
    data = request.json
    feats = [float(data['income']), float(data['loan_amount']), float(data['credit_score'])]
    pred = model.predict([feats])[0]
    actual = data.get('actual')
    
    PREDICTION_COUNT.inc()
    push_to_sidecar(feats, pred, actual)
    return jsonify({"prediction": int(pred)})

@app.route('/', methods=['GET', 'POST'])
def index():
    REQUEST_COUNT.inc()
    if request.method == 'POST':
        feats = [float(request.form['income']), float(request.form['loan']), float(request.form['credit'])]
        pred = model.predict([feats])[0]
        actual = request.form.get('actual')
        
        PREDICTION_COUNT.inc()
        push_to_sidecar(feats, pred, actual)
        return render_template('index.html', prediction=pred)
    return render_template('index.html')

# Expose /metrics for Prometheus
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {'/metrics': make_wsgi_app()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)