from flask import Flask, render_template, request
import joblib
import pandas as pd
from sdk.exporter import MLExporter, LATENCY, PREDICTION_VAL

monitor = MLExporter(port=8000)
app = Flask(__name__)
model = joblib.load('models/iris_model.joblib')

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    monitor.track_health()
    with LATENCY.time():
        data = [float(request.form['s_length']), float(request.form['s_width']),
                float(request.form['p_length']), float(request.form['p_width'])]
        df = pd.DataFrame([data], columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
        pred = int(model.predict(df)[0])
        PREDICTION_VAL.set(pred)
        monitor.check_drift(df)
        return render_template('index.html', prediction=pred)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)