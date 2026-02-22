# model_setup.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 1. Create a dummy baseline dataset
data = {
    'income': [5000, 4000, 7000, 3000, 8000],
    'loan_amount': [20000, 15000, 30000, 10000, 40000],
    'credit_score': [1, 1, 1, 0, 1],
    'target': [1, 1, 1, 0, 1]
}
df = pd.DataFrame(data)
df.to_csv('data/baseline_data.csv', index=False)

# 2. Train and save the model
X = df.drop('target', axis=1)
y = df['target']
model = RandomForestClassifier().fit(X, y)
joblib.dump(model, 'models/loan_model.joblib')

print("Setup Complete: Artifacts saved in data/ and models/ folders.")