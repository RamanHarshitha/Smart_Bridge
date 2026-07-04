import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'opticrop-data.json')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'image_disease_model.joblib')

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

feature_rows = []

def make_profile(label, crop, green, brown, dark, yellow, edge, brightness, contrast):
    return {
        'crop': crop,
        'green_ratio': green,
        'brown_ratio': brown,
        'dark_ratio': dark,
        'yellow_ratio': yellow,
        'edge_density': edge,
        'brightness': brightness,
        'contrast': contrast,
        'disease': label,
    }

profiles = {
    'Rice Blast': {'crop': 'Rice', 'green': 0.50, 'brown': 0.24, 'dark': 0.16, 'yellow': 0.06, 'edge': 0.20, 'brightness': 0.44, 'contrast': 0.54},
    'Brown Spot': {'crop': 'Rice', 'green': 0.52, 'brown': 0.18, 'dark': 0.10, 'yellow': 0.12, 'edge': 0.16, 'brightness': 0.47, 'contrast': 0.43},
    'Rust': {'crop': 'Wheat', 'green': 0.45, 'brown': 0.10, 'dark': 0.07, 'yellow': 0.22, 'edge': 0.14, 'brightness': 0.51, 'contrast': 0.48},
    'Northern Leaf Blight': {'crop': 'Maize', 'green': 0.56, 'brown': 0.12, 'dark': 0.12, 'yellow': 0.08, 'edge': 0.18, 'brightness': 0.46, 'contrast': 0.50},
    'Late Blight': {'crop': 'Tomato', 'green': 0.38, 'brown': 0.20, 'dark': 0.24, 'yellow': 0.05, 'edge': 0.26, 'brightness': 0.35, 'contrast': 0.61},
    'Early Blight': {'crop': 'Potato', 'green': 0.40, 'brown': 0.22, 'dark': 0.20, 'yellow': 0.04, 'edge': 0.24, 'brightness': 0.37, 'contrast': 0.58},
    'Cercospora Leaf Blight': {'crop': 'Soybean', 'green': 0.50, 'brown': 0.17, 'dark': 0.11, 'yellow': 0.07, 'edge': 0.16, 'brightness': 0.42, 'contrast': 0.46},
    'Black Sigatoka': {'crop': 'Banana', 'green': 0.44, 'brown': 0.16, 'dark': 0.24, 'yellow': 0.05, 'edge': 0.22, 'brightness': 0.40, 'contrast': 0.60},
}

rng = np.random.default_rng(42)
for disease, base in profiles.items():
    for _ in range(200):
        feature_rows.append(make_profile(
            disease,
            base['crop'],
            round(float(np.clip(base['green'] + rng.uniform(-0.18, 0.18), 0.05, 0.95)), 3),
            round(float(np.clip(base['brown'] + rng.uniform(-0.15, 0.15), 0.0, 0.85)), 3),
            round(float(np.clip(base['dark'] + rng.uniform(-0.15, 0.15), 0.0, 0.85)), 3),
            round(float(np.clip(base['yellow'] + rng.uniform(-0.12, 0.12), 0.0, 0.85)), 3),
            round(float(np.clip(base['edge'] + rng.uniform(-0.10, 0.10), 0.05, 0.85)), 3),
            round(float(np.clip(base['brightness'] + rng.uniform(-0.15, 0.15), 0.05, 0.95)), 3),
            round(float(np.clip(base['contrast'] + rng.uniform(-0.15, 0.15), 0.05, 0.95)), 3),
        ))


frame = pd.DataFrame(feature_rows)
feature_columns = ['crop', 'green_ratio', 'brown_ratio', 'dark_ratio', 'yellow_ratio', 'edge_density', 'brightness', 'contrast']

crop_encoder = LabelEncoder()
frame['crop'] = crop_encoder.fit_transform(frame['crop'])
disease_encoder = LabelEncoder()
y = disease_encoder.fit_transform(frame['disease'])
X = frame[feature_columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=250, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print('Image disease model accuracy:', round(accuracy_score(y_test, predictions), 4))
print(classification_report(y_test, predictions, labels=sorted(set(y_test)), target_names=disease_encoder.inverse_transform(sorted(set(y_test))), zero_division=0))

joblib.dump({
    'model': model,
    'crop_encoder': crop_encoder,
    'disease_encoder': disease_encoder,
    'feature_columns': feature_columns,
}, MODEL_PATH)
print('Saved image disease model to', MODEL_PATH)
