import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
CSV_DATASET_PATH = os.path.join(ROOT, 'data', 'Crop_recommendation.csv')
MODEL_PATH = os.path.join(ROOT, 'models', 'crop_model.joblib')
PLOT_PATH = os.path.join(ROOT, 'models', 'confusion_matrix.png')
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

if not os.path.exists(CSV_DATASET_PATH):
    raise FileNotFoundError(f'Crop recommendation CSV not found at {CSV_DATASET_PATH}')

frame = pd.read_csv(CSV_DATASET_PATH)
frame = frame.rename(columns={
    'N': 'nitrogen',
    'P': 'phosphorous',
    'K': 'potassium',
    'label': 'crop',
    'ph': 'ph',
})

frame = frame[['nitrogen', 'phosphorous', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall', 'crop']].copy()

location_encoder = LabelEncoder()
season_encoder = LabelEncoder()
frame['location'] = 'default'
frame['season'] = 'kharif'
frame['location'] = location_encoder.fit_transform(frame['location'])
frame['season'] = season_encoder.fit_transform(frame['season'])

features = ['nitrogen', 'phosphorous', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall', 'location', 'season']
X = frame[features]
y = frame['crop']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

split_kwargs = {'test_size': 0.2, 'random_state': 42}
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, **split_kwargs)
print('Training split created successfully')

models = {
    'random_forest': RandomForestClassifier(n_estimators=350, random_state=42, class_weight='balanced_subsample'),
    'decision_tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'svc': make_pipeline(StandardScaler(), SVC(probability=True, random_state=42, class_weight='balanced')),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f'{name} accuracy:', round(accuracy_score(y_test, predictions), 4))

joblib.dump({
    'models': models,
    'model_names': list(models.keys()),
    'encoder': le,
    'location_encoder': location_encoder,
    'season_encoder': season_encoder,
    'features': features,
}, MODEL_PATH)

cm = pd.crosstab(y_test, models['random_forest'].predict(X_test))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title('Crop Prediction Confusion Matrix')
plt.xlabel('Predicted Crop')
plt.ylabel('Actual Crop')
plt.tight_layout()
plt.savefig(PLOT_PATH)
print('Saved model to', MODEL_PATH)
print('Saved plot to', PLOT_PATH)
