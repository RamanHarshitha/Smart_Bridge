import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'opticrop-data.json')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'disease_model.joblib')
PLOT_PATH = os.path.join(os.path.dirname(__file__), 'disease_confusion_matrix.png')

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

base_records = []
for entry in dataset['diseases']:
    base_records.append({
        'crop': entry['crop'],
        'temperature': float(entry['temperature']),
        'humidity': float(entry['humidity']),
        'ph': float(entry['ph']),
        'disease': entry['disease'],
        'symptoms': entry['symptoms']
    })

rng = np.random.default_rng(42)
augmented_rows = []
for record in base_records:
    for _ in range(10):
        augmented_rows.append({
            'crop': record['crop'],
            'temperature': round(min(45, max(8, record['temperature'] + rng.uniform(-2.5, 2.5))), 1),
            'humidity': round(min(98, max(40, record['humidity'] + rng.uniform(-8, 8))), 1),
            'ph': round(min(8.5, max(4.5, record['ph'] + rng.uniform(-0.35, 0.35))), 1),
            'disease': record['disease'],
            'symptoms': record['symptoms']
        })

symptom_columns = sorted({symptom for row in augmented_rows for symptom in row['symptoms']})
rows = []
for row in augmented_rows:
    feature_row = {
        'crop': row['crop'],
        'temperature': row['temperature'],
        'humidity': row['humidity'],
        'ph': row['ph'],
        'disease': row['disease']
    }
    for symptom in symptom_columns:
        feature_row[symptom] = int(symptom in row['symptoms'])
    rows.append(feature_row)

frame = pd.DataFrame(rows)
feature_columns = ['crop', 'temperature', 'humidity', 'ph'] + symptom_columns
frame = frame[feature_columns + ['disease']]

crop_encoder = LabelEncoder()
frame['crop'] = crop_encoder.fit_transform(frame['crop'])
disease_encoder = LabelEncoder()
y = disease_encoder.fit_transform(frame['disease'])
X = frame[feature_columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = RandomForestClassifier(n_estimators=250, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print('Disease model accuracy:', round(accuracy_score(y_test, predictions), 4))
print(classification_report(y_test, predictions, labels=sorted(set(y_test)), target_names=disease_encoder.inverse_transform(sorted(set(y_test))), zero_division=0))

joblib.dump({
    'model': model,
    'crop_encoder': crop_encoder,
    'disease_encoder': disease_encoder,
    'feature_columns': feature_columns,
    'symptom_columns': symptom_columns
}, MODEL_PATH)

cm = pd.crosstab(y_test, predictions)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title('Disease Classification Confusion Matrix')
plt.xlabel('Predicted Disease')
plt.ylabel('Actual Disease')
plt.tight_layout()
plt.savefig(PLOT_PATH)
print('Saved disease model to', MODEL_PATH)
print('Saved disease plot to', PLOT_PATH)
