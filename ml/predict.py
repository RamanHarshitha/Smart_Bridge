import json
import os
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'crop_model.joblib')

with open(os.path.join(os.path.dirname(__file__), '..', 'opticrop-data.json'), 'r', encoding='utf-8') as f:
    dataset = json.load(f)

bundle = joblib.load(MODEL_PATH)
encoder = bundle['encoder']
features = bundle['features']
location_encoder = bundle.get('location_encoder')
season_encoder = bundle.get('season_encoder')
model_names = bundle.get('model_names', ['random_forest'])
models = bundle.get('models', {})
available_models = [models[name] for name in model_names if name in models]
if not available_models and 'model' in bundle:
    available_models = [bundle['model']]

sample = {
    'nitrogen': 110,
    'phosphorous': 55,
    'potassium': 70,
    'temperature': 26,
    'humidity': 72,
    'ph': 6.4,
    'rainfall': 95,
    'location': 0,
    'season': 0,
}

input_frame = pd.DataFrame([sample])[features]

probability_matrix = np.zeros((len(encoder.classes_),), dtype=float)
for model in available_models:
    probs = model.predict_proba(input_frame)
    if probs.ndim == 2:
        probs = probs[0]
    probability_matrix += np.array(probs, dtype=float)

if probability_matrix.sum() > 0:
    probability_matrix /= probability_matrix.sum()

prediction = int(np.argmax(probability_matrix))
print('Predicted crop:', encoder.inverse_transform([prediction])[0])
print('Confidence:', f'{probability_matrix.max() * 100:.2f}%')
