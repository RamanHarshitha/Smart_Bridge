import io
import json
import os
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import joblib
import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(__file__)
WEB_ROOT = os.path.join(ROOT, 'web')
DATA_ROOT = os.path.join(ROOT, 'data')
DATASET_PATH = os.path.join(DATA_ROOT, 'opticrop-data.json')
CROP_CSV_PATH = os.path.join(DATA_ROOT, 'Crop_recommendation.csv')
KAGGLE_DISEASE_ROOT = os.path.join(DATA_ROOT, 'datasets', 'plant_disease')
CROP_MODEL_PATH = os.path.join(ROOT, 'models', 'crop_model.joblib')
DISEASE_MODEL_PATH = os.path.join(ROOT, 'models', 'disease_model.joblib')
IMAGE_DISEASE_MODEL_PATH = os.path.join(ROOT, 'models', 'image_disease_model.joblib')
IMAGE_DISEASE_KAGGLE_MODEL_PATH = os.path.join(ROOT, 'models', 'image_disease_model_keras.h5')
IMAGE_DISEASE_KAGGLE_CLASSES_PATH = os.path.join(ROOT, 'models', 'image_disease_model_classes.json')
KAGGLE_IMAGE_MODEL = None
KAGGLE_IMAGE_CLASSES = None

CROP_DISEASE_MAPPING = {
    'Rice': ['Rice Blast', 'Brown Spot'],
    'Wheat': ['Rust'],
    'Maize': ['Northern Leaf Blight'],
    'Tomato': ['Late Blight'],
    'Potato': ['Early Blight'],
    'Soybean': ['Cercospora Leaf Blight'],
    'Banana': ['Black Sigatoka']
}



def get_fertilizer_recommendation(payload, crop_name):
    crop = (crop_name or 'General').strip()
    nitrogen = float(payload.get('nitrogen', 0))
    phosphorous = float(payload.get('phosphorous', 0))
    potassium = float(payload.get('potassium', 0))
    ph = float(payload.get('ph', 6.5))

    base_recommendations = {
        'Rice': [
            {'nutrient': 'Nitrogen', 'target': 'Apply split nitrogen dosing to support tillering and grain filling.', 'priority': 'High'},
            {'nutrient': 'Phosphorus', 'target': 'Use phosphorus-rich basal nutrition for root strength.', 'priority': 'Medium'},
            {'nutrient': 'Potassium', 'target': 'Support stress tolerance and grain quality with balanced potash.', 'priority': 'High'},
        ],
        'Wheat': [
            {'nutrient': 'Nitrogen', 'target': 'Use moderate nitrogen to support early growth without excess lodging.', 'priority': 'Medium'},
            {'nutrient': 'Phosphorus', 'target': 'Keep phosphorus steady for healthy roots and early vigor.', 'priority': 'Medium'},
            {'nutrient': 'Potassium', 'target': 'Maintain potash to support grain fill under cooler conditions.', 'priority': 'Medium'},
        ],
        'Maize': [
            {'nutrient': 'Nitrogen', 'target': 'Use a strong nitrogen program during vegetative growth.', 'priority': 'High'},
            {'nutrient': 'Phosphorus', 'target': 'Support root establishment with basal phosphorus.', 'priority': 'High'},
            {'nutrient': 'Potassium', 'target': 'Ensure ample potassium for stalk strength and yield.', 'priority': 'High'},
        ],
        'Tomato': [
            {'nutrient': 'Nitrogen', 'target': 'Keep nitrogen balanced for leafy growth and fruiting.', 'priority': 'Medium'},
            {'nutrient': 'Phosphorus', 'target': 'Boost phosphorus early for root and flowering support.', 'priority': 'Medium'},
            {'nutrient': 'Potassium', 'target': 'Prioritize potassium for fruit size and disease tolerance.', 'priority': 'High'},
        ],
        'Potato': [
            {'nutrient': 'Nitrogen', 'target': 'Use controlled nitrogen to avoid excess canopy growth.', 'priority': 'Medium'},
            {'nutrient': 'Phosphorus', 'target': 'Support tuber initiation with steady phosphorus.', 'priority': 'Medium'},
            {'nutrient': 'Potassium', 'target': 'Maintain strong potassium supply for tuber bulking.', 'priority': 'High'},
        ],
        'Soybean': [
            {'nutrient': 'Nitrogen', 'target': 'Avoid heavy nitrogen and rely on biological fixation where possible.', 'priority': 'Low'},
            {'nutrient': 'Phosphorus', 'target': 'Maintain phosphorus for healthy root nodulation.', 'priority': 'High'},
            {'nutrient': 'Potassium', 'target': 'Keep potassium moderate to support pod formation.', 'priority': 'Medium'},
        ],
        'Banana': [
            {'nutrient': 'Nitrogen', 'target': 'Use regular nitrogen applications during active leaf growth.', 'priority': 'High'},
            {'nutrient': 'Phosphorus', 'target': 'Support root growth with phased phosphorus applications.', 'priority': 'Medium'},
            {'nutrient': 'Potassium', 'target': 'Prioritize potassium for fruit quality and bunch weight.', 'priority': 'High'},
        ],
    }

    soil_note = 'Soil pH is within a good range.'
    if ph < 5.8:
        soil_note = 'The soil is slightly acidic; add lime or organic matter to raise pH gradually.'
    elif ph > 7.4:
        soil_note = 'The soil is slightly alkaline; use compost and acidifying amendments where possible.'

    recommendation = base_recommendations.get(crop, [
        {'nutrient': 'Nitrogen', 'target': 'Apply moderate nitrogen to match current crop vigor.', 'priority': 'Medium'},
        {'nutrient': 'Phosphorus', 'target': 'Keep phosphorus balanced for root health.', 'priority': 'Medium'},
        {'nutrient': 'Potassium', 'target': 'Support resilience with balanced potash.', 'priority': 'Medium'},
    ])

    if nitrogen > 160:
        recommendation[0]['target'] = 'Reduce nitrogen intensity and switch to split-dosed applications to prevent excess vegetative growth.'
    elif nitrogen < 70:
        recommendation[0]['target'] = 'Increase nitrogen carefully with a split-applications plan to improve early growth.'

    if phosphorous < 35:
        recommendation[1]['target'] = 'Raise phosphorus through basal fertilization to strengthen root development.'

    if potassium < 55:
        recommendation[2]['target'] = 'Increase potassium to improve stress tolerance and yield stability.'

    return {
        'crop': crop,
        'summary': f'{crop} is best supported by a balanced fertilization plan that matches current soil and climate conditions.',
        'recommendation': recommendation,
        'soil_note': soil_note,
        'nutrient_balance': {
            'nitrogen': 'Low' if nitrogen < 80 else 'High' if nitrogen > 150 else 'Balanced',
            'phosphorous': 'Low' if phosphorous < 40 else 'High' if phosphorous > 85 else 'Balanced',
            'potassium': 'Low' if potassium < 60 else 'High' if potassium > 140 else 'Balanced',
        },
    }


VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}


def normalize_crop_name(name):
    normalized = (name or '').strip().replace('_', ' ').replace(',', ' ')
    normalized = normalized.replace('(', ' ').replace(')', ' ')
    normalized = ' '.join(normalized.split())
    if normalized.lower() in {'corn maize', 'corn'}:
        return 'Maize'
    if normalized.lower() == 'potato':
        return 'Potato'
    if normalized.lower() == 'tomato':
        return 'Tomato'
    if normalized.lower() == 'soybean':
        return 'Soybean'
    if normalized.lower() == 'banana':
        return 'Banana'
    if normalized.lower() == 'rice':
        return 'Rice'
    if normalized.lower() == 'wheat':
        return 'Wheat'
    return normalized.title()


def discover_kaggle_disease_profiles():
    if not os.path.isdir(KAGGLE_DISEASE_ROOT):
        return []

    profiles = []
    for root, _, files in os.walk(KAGGLE_DISEASE_ROOT):
        image_files = [file for file in files if os.path.splitext(file)[1].lower() in VALID_IMAGE_EXTENSIONS]
        if not image_files:
            continue

        class_name = os.path.basename(root)
        if '___' not in class_name and class_name.lower() in {'train', 'valid'}:
            continue

        crop_name = normalize_crop_name(class_name.split('___')[0])
        disease_name = class_name.split('___')[1] if '___' in class_name else class_name
        disease_name = disease_name.replace('_', ' ').replace('  ', ' ').strip().title()

        if not crop_name or crop_name.lower() in {'train', 'valid'}:
            continue

        profiles.append({
            'crop': crop_name,
            'disease': disease_name or 'Leaf disease',
            'temperature': 26.5,
            'humidity': 78.0,
            'ph': 6.7,
            'symptoms': ['Leaf discoloration', 'Lesions', 'Wilting']
        })
    return profiles


def load_dataset():
    if os.path.exists(CROP_CSV_PATH):
        frame = pd.read_csv(CROP_CSV_PATH)
        frame = frame.rename(columns={
            'N': 'nitrogen',
            'P': 'phosphorous',
            'K': 'potassium',
            'label': 'crop',
            'ph': 'ph',
        })
        frame = frame[['nitrogen', 'phosphorous', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall', 'crop']].copy()
        frame['crop'] = frame['crop'].astype(str)
        locations = [
            {'id': 'north-india', 'name': 'North India'},
            {'id': 'south-india', 'name': 'South India'},
            {'id': 'central-india', 'name': 'Central India'},
        ]
        seasons = [
            {'id': 'kharif', 'name': 'Kharif', 'months': 'Jun-Oct'},
            {'id': 'rabi', 'name': 'Rabi', 'months': 'Nov-Mar'},
            {'id': 'zaid', 'name': 'Zaid', 'months': 'Apr-Jun'},
        ]
        records = []
        for index, row in frame.iterrows():
            location = locations[index % len(locations)]['id']
            season = seasons[index % len(seasons)]['id']
            records.append({
                'crop': row['crop'],
                'nitrogen': float(row['nitrogen']),
                'phosphorous': float(row['phosphorous']),
                'potassium': float(row['potassium']),
                'temperature': float(row['temperature']),
                'humidity': float(row['humidity']),
                'ph': float(row['ph']),
                'rainfall': float(row['rainfall']),
                'location': location,
                'season': season,
                'description': f"{row['crop']} profile from the Kaggle crop recommendation dataset.",
                'yield': 'High',
                'strategy': 'Monitor field inputs and seasonal rainfall closely.'
            })
        disease_profiles = discover_kaggle_disease_profiles()
        if not disease_profiles:
            disease_profiles = [
                {'crop': 'Rice', 'disease': 'Rice Blast', 'temperature': 27.0, 'humidity': 80.0, 'ph': 6.5, 'symptoms': ['Leaf spotting', 'Lesions']},
                {'crop': 'Tomato', 'disease': 'Late Blight', 'temperature': 24.0, 'humidity': 82.0, 'ph': 6.3, 'symptoms': ['Water-soaked lesions', 'Leaf decay']},
                {'crop': 'Potato', 'disease': 'Early Blight', 'temperature': 24.5, 'humidity': 75.0, 'ph': 6.2, 'symptoms': ['Brown spots', 'Yellowing']},
                {'crop': 'Maize', 'disease': 'Northern Leaf Blight', 'temperature': 25.0, 'humidity': 70.0, 'ph': 6.8, 'symptoms': ['Leaf streaks', 'Necrosis']},
            ]
        return {
            'crops': records,
            'locations': locations,
            'seasons': seasons,
            'diseases': disease_profiles,
        }

    with open(DATASET_PATH, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def score_metric(value, minimum, maximum):
    ideal_mid = (minimum + maximum) / 2
    half_range = (maximum - minimum) / 2 or 1
    if minimum <= value <= maximum:
        return 1.0
    deviation = abs(value - ideal_mid) - half_range
    return max(0.15, 1 - deviation / (half_range * 1.5))


def get_crop_records(crop_name, location=None, season=None):
    dataset = load_dataset()
    records = [entry for entry in dataset['crops'] if entry['crop'] == crop_name]
    if location:
        records = [entry for entry in records if entry.get('location') == location] or records
    if season:
        records = [entry for entry in records if entry.get('season') == season] or records
    return records


def evaluate_crop_suitability(payload):
    crop_name = (payload.get('crop') or '').strip()
    if not crop_name:
        return {'error': 'Crop name is required'}

    location = payload.get('location') or None
    season = payload.get('season') or None
    records = get_crop_records(crop_name, location, season)
    if not records:
        return {'error': f'No profile found for crop: {crop_name}'}

    values = {
        'nitrogen': float(payload.get('nitrogen', 0)),
        'phosphorous': float(payload.get('phosphorous', 0)),
        'potassium': float(payload.get('potassium', 0)),
        'temperature': float(payload.get('temperature', 0)),
        'humidity': float(payload.get('humidity', 0)),
        'ph': float(payload.get('ph', 6.5)),
        'rainfall': float(payload.get('rainfall', 0)),
    }

    metric_defs = [
        ('Nitrogen', 'nitrogen', 0.2),
        ('Phosphorous', 'phosphorous', 0.16),
        ('Potassium', 'potassium', 0.16),
        ('Temperature', 'temperature', 0.14),
        ('Humidity', 'humidity', 0.13),
        ('pH', 'ph', 0.1),
        ('Rainfall', 'rainfall', 0.11),
    ]

    metrics = []
    remediation = []
    for label, key, weight in metric_defs:
        field_values = [entry[key] for entry in records]
        minimum = min(field_values) * (0.9 if key != 'ph' else 0.95)
        maximum = max(field_values) * (1.1 if key != 'ph' else 1.05)
        score = score_metric(values[key], minimum, maximum)
        status = 'Optimal' if score >= 0.85 else 'Acceptable' if score >= 0.65 else 'Suboptimal' if score >= 0.45 else 'Poor'
        ideal_mid = round((minimum + maximum) / 2, 1)
        metrics.append({
            'label': label,
            'key': key,
            'score': round(score, 3),
            'status': status,
            'ideal_range': [round(minimum, 1), round(maximum, 1)],
            'current_value': values[key],
            'ideal_mid': ideal_mid,
            'weight': weight,
        })
        if score < 0.65:
            direction = 'increase' if values[key] < ideal_mid else 'decrease'
            delta = abs(values[key] - ideal_mid)
            remediation.append({
                'factor': label,
                'action': f'{direction.capitalize()} {label.lower()} toward ~{ideal_mid}',
                'detail': f'Current {label.lower()} is {values[key]}; target range is {round(minimum, 1)}–{round(maximum, 1)}.',
                'priority': 'High' if score < 0.45 else 'Medium',
            })

    weighted_score = sum(item['score'] * item['weight'] for item in metrics)
    suitability = (
        'Excellent' if weighted_score >= 0.85 else
        'Strong' if weighted_score >= 0.7 else
        'Moderate' if weighted_score >= 0.55 else
        'Poor'
    )
    productivity = (
        'Very High' if weighted_score >= 0.85 else
        'High' if weighted_score >= 0.7 else
        'Moderate' if weighted_score >= 0.55 else
        'Low'
    )
    profile = records[0]
    compatible = weighted_score >= 0.55

    location_note = ''
    if location:
        location_names = {entry['id']: entry['name'] for entry in load_dataset().get('locations', [])}
        location_note = f' Evaluated for {location_names.get(location, location)}.'
    if season:
        season_names = {entry['id']: entry['name'] for entry in load_dataset().get('seasons', [])}
        location_note += f' Season context: {season_names.get(season, season)}.'

    return {
        'crop': crop_name,
        'compatible': compatible,
        'suitability': suitability,
        'productivity_potential': productivity,
        'overall_score': round(weighted_score * 100, 1),
        'description': profile.get('description', ''),
        'yield_outlook': profile.get('yield', 'Moderate'),
        'strategy': profile.get('strategy', ''),
        'metrics': metrics,
        'remediation': remediation,
        'summary': (
            f'{crop_name} shows {suitability.lower()} compatibility ({round(weighted_score * 100, 1)}% fit) '
            f'with productivity potential rated {productivity.lower()}.{location_note}'
        ),
        'fertilizer': get_fertilizer_recommendation(payload, crop_name),
    }


def fetch_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'OptiCrop/1.0'})
    with urllib.request.urlopen(request, timeout=12) as response:
        return json.loads(response.read().decode('utf-8'))


def get_weather_data(city=None, latitude=None, longitude=None):
    if city:
        query = urllib.parse.urlencode({'name': city, 'count': 1, 'language': 'en', 'format': 'json'})
        geo = fetch_json(f'https://geocoding-api.open-meteo.com/v1/search?{query}')
        results = geo.get('results') or []
        if not results:
            return {'error': f'Location not found: {city}'}
        place = results[0]
        latitude = place['latitude']
        longitude = place['longitude']
        location_label = ', '.join(part for part in [place.get('name'), place.get('admin1'), place.get('country')] if part)
    elif latitude is not None and longitude is not None:
        location_label = f'{latitude:.2f}, {longitude:.2f}'
    else:
        return {'error': 'Provide city or latitude/longitude'}

    params = urllib.parse.urlencode({
        'latitude': latitude,
        'longitude': longitude,
        'current': 'temperature_2m,relative_humidity_2m',
        'daily': 'precipitation_sum',
        'past_days': 30,
        'forecast_days': 1,
        'timezone': 'auto',
    })
    weather = fetch_json(f'https://api.open-meteo.com/v1/forecast?{params}')
    current = weather.get('current', {})
    daily = weather.get('daily', {})
    rainfall_total = sum(value or 0 for value in daily.get('precipitation_sum', []))

    return {
        'location': location_label,
        'latitude': latitude,
        'longitude': longitude,
        'temperature': round(float(current.get('temperature_2m', 25)), 1),
        'humidity': round(float(current.get('relative_humidity_2m', 65)), 1),
        'rainfall': round(float(rainfall_total), 1),
        'rainfall_period': '30-day total (mm)',
        'source': 'Open-Meteo',
    }


def load_keras_image_disease_model():
    global KAGGLE_IMAGE_MODEL, KAGGLE_IMAGE_CLASSES
    if KAGGLE_IMAGE_MODEL is not None and KAGGLE_IMAGE_CLASSES is not None:
        return KAGGLE_IMAGE_MODEL, KAGGLE_IMAGE_CLASSES
    if not os.path.exists(IMAGE_DISEASE_KAGGLE_MODEL_PATH) or not os.path.exists(IMAGE_DISEASE_KAGGLE_CLASSES_PATH):
        return None, None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(IMAGE_DISEASE_KAGGLE_MODEL_PATH)
        with open(IMAGE_DISEASE_KAGGLE_CLASSES_PATH, 'r', encoding='utf-8') as fh:
            classes = json.load(fh)
        KAGGLE_IMAGE_MODEL = model
        KAGGLE_IMAGE_CLASSES = classes
        return model, classes
    except Exception as exc:
        print('Unable to load Kaggle image disease model:', exc)
        return None, None


def predict_image_disease_with_keras(image, crop_name):
    model, classes = load_keras_image_disease_model()
    if model is None or classes is None:
        return None
    import tensorflow as tf
    image = image.convert('RGB').resize((224, 224))
    array = tf.keras.preprocessing.image.img_to_array(image)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    array = np.expand_dims(array, axis=0)
    probabilities = model.predict(array, verbose=0)[0]
    top_idx = int(np.argmax(probabilities))
    predicted = classes[top_idx]
    confidence = round(float(np.max(probabilities) * 100), 2)
    return {
        'predicted_disease': predicted,
        'confidence': confidence,
        'probabilities': {classes[i]: float(probabilities[i]) for i in range(len(classes))}
    }


def build_analytics_summary():
    dataset = load_dataset()
    crops = dataset['crops']
    crop_names = sorted({entry['crop'] for entry in crops})
    locations = dataset.get('locations', [])
    seasons = dataset.get('seasons', [])

    by_crop = []
    for crop_name in crop_names:
        rows = [entry for entry in crops if entry['crop'] == crop_name]
        by_crop.append({
            'crop': crop_name,
            'count': len(rows),
            'avg_nitrogen': round(sum(entry['nitrogen'] for entry in rows) / len(rows), 1),
            'avg_phosphorous': round(sum(entry['phosphorous'] for entry in rows) / len(rows), 1),
            'avg_potassium': round(sum(entry['potassium'] for entry in rows) / len(rows), 1),
            'avg_temperature': round(sum(entry['temperature'] for entry in rows) / len(rows), 1),
            'avg_humidity': round(sum(entry['humidity'] for entry in rows) / len(rows), 1),
            'avg_ph': round(sum(entry['ph'] for entry in rows) / len(rows), 2),
            'avg_rainfall': round(sum(entry['rainfall'] for entry in rows) / len(rows), 1),
            'yield_levels': sorted({entry.get('yield', 'Moderate') for entry in rows}),
        })

    by_location = []
    for location in locations:
        rows = [entry for entry in crops if entry.get('location') == location['id']]
        if not rows:
            continue
        by_location.append({
            'location': location['name'],
            'location_id': location['id'],
            'count': len(rows),
            'avg_temperature': round(sum(entry['temperature'] for entry in rows) / len(rows), 1),
            'avg_rainfall': round(sum(entry['rainfall'] for entry in rows) / len(rows), 1),
            'top_crops': sorted({entry['crop'] for entry in rows}),
        })

    by_season = []
    for season in seasons:
        rows = [entry for entry in crops if entry.get('season') == season['id']]
        if not rows:
            continue
        by_season.append({
            'season': season['name'],
            'season_id': season['id'],
            'count': len(rows),
            'avg_temperature': round(sum(entry['temperature'] for entry in rows) / len(rows), 1),
            'avg_rainfall': round(sum(entry['rainfall'] for entry in rows) / len(rows), 1),
            'top_crops': sorted({entry['crop'] for entry in rows}),
        })

    scatter_points = [{
        'crop': entry['crop'],
        'location': entry.get('location'),
        'season': entry.get('season'),
        'temperature': entry['temperature'],
        'humidity': entry['humidity'],
        'rainfall': entry['rainfall'],
        'nitrogen': entry['nitrogen'],
        'phosphorous': entry['phosphorous'],
        'potassium': entry['potassium'],
        'ph': entry['ph'],
        'yield': entry.get('yield', 'Moderate'),
    } for entry in crops]

    return {
        'totals': {
            'records': len(crops),
            'crops': len(crop_names),
            'locations': len(locations),
            'seasons': len(seasons),
            'diseases': len(dataset.get('diseases', [])),
        },
        'locations': locations,
        'seasons': seasons,
        'by_crop': by_crop,
        'by_location': by_location,
        'by_season': by_season,
        'scatter_points': scatter_points,
    }


class OptiCropServer(ThreadingHTTPServer):
    allow_reuse_address = True


ALLOWED_EXTENSIONS = {'.html', '.css', '.js', '.json', '.png', '.jpg', '.ico', '.svg', '.webp'}


def send_404(handler):
    handler.send_response(404)
    handler.send_header('Content-Type', 'text/html; charset=utf-8')
    handler.end_headers()
    handler.wfile.write('<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>404 | OptiCrop</title><link rel="stylesheet" href="/styles.css"></head><body data-page="404"><header class="topbar"><a class="brand" href="/"><span class="brand-mark">O</span><span><strong>OptiCrop</strong><small>AgriSens</small></span></a></header><main class="panel" style="text-align:center;padding-top:80px;"><h1>404 &mdash; Page not found</h1><p class="muted" style="margin:16px 0 28px;">The page you requested does not exist.</p><a class="btn primary" href="/">Back to home</a></main></body></html>'.encode('utf-8'))



def safe_resolve(path):
    """Resolve a URL path to a file path, preventing directory traversal."""
    cleaned = urllib.parse.urlparse(path).path.lstrip('/')
    if cleaned.startswith('data/'):
        base_root = DATA_ROOT
        relative_path = cleaned[len('data/'):]
        file_path = os.path.realpath(os.path.join(base_root, relative_path))
        if not file_path.startswith(os.path.realpath(base_root)):
            return None
    else:
        file_path = os.path.realpath(os.path.join(WEB_ROOT, cleaned))
        if not file_path.startswith(os.path.realpath(WEB_ROOT)):
            return None
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return None
    if os.path.isfile(file_path):
        return file_path
    return None


CONTENT_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.ico': 'image/x-icon',
    '.svg': 'image/svg+xml',
    '.webp': 'image/webp',
}


class OptiCropHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            with open(os.path.join(WEB_ROOT, 'index.html'), 'rb') as fh:
                self.wfile.write(fh.read())
            return

        if self.path.startswith('/api/weather'):
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            city = params.get('city', [None])[0]
            latitude = params.get('lat', [None])[0]
            longitude = params.get('lon', [None])[0]
            try:
                latitude = float(latitude) if latitude is not None else None
                longitude = float(longitude) if longitude is not None else None
                payload = get_weather_data(city=city, latitude=latitude, longitude=longitude)
                status = 200 if 'error' not in payload else 404
            except Exception as exc:
                payload = {'error': str(exc)}
                status = 502
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode('utf-8'))
            return

        if self.path.startswith('/api/analytics'):
            try:
                payload = build_analytics_summary()
                status = 200
            except Exception as exc:
                payload = {'error': str(exc)}
                status = 500
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode('utf-8'))
            return

        if self.path == '/data/opticrop-data.json':
            payload = load_dataset()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode('utf-8'))
            return

        file_path = safe_resolve(self.path)
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()
            self.send_response(200)
            self.send_header('Content-Type', CONTENT_TYPES.get(ext, 'application/octet-stream'))
            self.end_headers()
            with open(file_path, 'rb') as fh:
                self.wfile.write(fh.read())
            return

        send_404(self)

    def do_HEAD(self):
        if self.path == '/' or self.path == '':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            return

        file_path = safe_resolve(self.path)
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()
            self.send_response(200)
            self.send_header('Content-Type', CONTENT_TYPES.get(ext, 'application/octet-stream'))
            self.end_headers()
            return

        self.send_response(404)
        self.end_headers()

    def _read_json_body(self):
        """Read and parse JSON from the request body. Returns None on failure."""
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode('utf-8')
            return json.loads(body)
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'error': f'Invalid JSON: {exc}'}).encode('utf-8'))
            return None

    def do_POST(self):
        if self.path == '/api/suitability':
            payload = self._read_json_body()
            if payload is None:
                return
            response = evaluate_crop_suitability(payload)
            status = 200 if 'error' not in response else 400
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            return

        if self.path == '/api/recommend':
            payload = self._read_json_body()
            if payload is None:
                return

            try:
                if not os.path.exists(CROP_MODEL_PATH):
                    raise FileNotFoundError('Crop model not trained yet')

                bundle = joblib.load(CROP_MODEL_PATH)
                encoder = bundle['encoder']
                features = bundle['features']
                model_names = bundle.get('model_names', ['random_forest'])
                models = bundle.get('models', {})
                available_models = [models[name] for name in model_names if name in models]
                if not available_models and 'model' in bundle:
                    available_models = [bundle['model']]

                model_payload = dict(payload)
                location_encoder = bundle.get('location_encoder')
                season_encoder = bundle.get('season_encoder')
                if location_encoder and payload.get('location'):
                    try:
                        model_payload['location'] = int(location_encoder.transform([payload['location']])[0])
                    except ValueError:
                        model_payload['location'] = 0
                elif 'location' in features:
                    model_payload['location'] = 0
                if season_encoder and payload.get('season'):
                    try:
                        model_payload['season'] = int(season_encoder.transform([payload['season']])[0])
                    except ValueError:
                        model_payload['season'] = 0
                elif 'season' in features:
                    model_payload['season'] = 0

                frame = pd.DataFrame([model_payload])[features]
                probability_matrix = np.zeros((len(encoder.classes_),), dtype=float)
                for model in available_models:
                    model_probs = model.predict_proba(frame)
                    if model_probs.ndim == 2:
                        model_probs = model_probs[0]

                    if hasattr(model, 'classes_') and len(model.classes_) != len(encoder.classes_):
                        aligned_probs = np.zeros_like(probability_matrix)
                        index_map = {label: idx for idx, label in enumerate(model.classes_)}
                        for global_idx, label in enumerate(encoder.classes_):
                            if label in index_map:
                                aligned_probs[global_idx] = float(model_probs[index_map[label]])
                    else:
                        aligned_probs = np.array(model_probs, dtype=float)

                    probability_matrix += aligned_probs

                if probability_matrix.sum() <= 0:
                    probability_matrix = np.ones_like(probability_matrix) / len(probability_matrix)
                else:
                    probability_matrix = probability_matrix / probability_matrix.sum()

                probabilities = probability_matrix
                prediction = int(np.argmax(probabilities))
                predicted_crop = encoder.inverse_transform([prediction])[0]
                confidence = round(float(probabilities.max() * 100), 2)
                ranked_candidates = [
                    {'crop': encoder.inverse_transform([idx])[0], 'probability': round(float(prob), 4)}
                    for idx, prob in sorted(zip(range(len(encoder.classes_)), probabilities), key=lambda item: item[1], reverse=True)[:10]
                ]

                response = {
                    'predicted_crop': predicted_crop,
                    'confidence': confidence,
                    'candidates': ranked_candidates,
                    'fertilizer': get_fertilizer_recommendation(payload, predicted_crop),
                }
            except Exception as exc:
                response = {
                    'predicted_crop': payload.get('crop') or 'Rice',
                    'confidence': 60.0,
                    'candidates': [
                        {'crop': 'Rice', 'probability': 0.42},
                        {'crop': 'Wheat', 'probability': 0.22},
                        {'crop': 'Maize', 'probability': 0.16},
                        {'crop': 'Potato', 'probability': 0.10},
                        {'crop': 'Tomato', 'probability': 0.08},
                    ],
                    'fertilizer': get_fertilizer_recommendation(payload, payload.get('crop') or 'Rice'),
                    'fallback': True,
                    'error': str(exc),
                }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            return

        if self.path == '/api/fertilizer':
            payload = self._read_json_body()
            if payload is None:
                return
            crop_name = payload.get('crop', 'General')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(get_fertilizer_recommendation(payload, crop_name)).encode('utf-8'))
            return

        if self.path == '/api/disease':
            payload = self._read_json_body()
            if payload is None:
                return

            if not os.path.exists(DISEASE_MODEL_PATH):
                self.send_response(503)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Disease model not trained yet'}).encode('utf-8'))
                return

            bundle = joblib.load(DISEASE_MODEL_PATH)
            model = bundle['model']
            crop_encoder = bundle['crop_encoder']
            disease_encoder = bundle['disease_encoder']
            feature_columns = bundle['feature_columns']
            symptom_columns = bundle['symptom_columns']

            encoded_payload = {
                'crop': crop_encoder.transform([payload['crop']])[0],
                'temperature': payload['temperature'],
                'humidity': payload['humidity'],
                'ph': payload['ph'],
            }
            for symptom in symptom_columns:
                encoded_payload[symptom] = int(symptom in payload.get('symptoms', []))

            frame = pd.DataFrame([encoded_payload])[feature_columns]
            probabilities = model.predict_proba(frame)[0]
            classes = disease_encoder.classes_

            crop_name = payload.get('crop', 'Rice')
            valid_diseases = CROP_DISEASE_MAPPING.get(crop_name, ['Rice Blast'])
            valid_probs = [(c, probabilities[idx]) for idx, c in enumerate(classes) if c in valid_diseases]

            if not valid_probs:
                predicted_disease = valid_diseases[0]
                confidence = 90.0
            else:
                best_disease, best_prob = max(valid_probs, key=lambda x: x[1])
                predicted_disease = best_disease
                total_valid_prob = sum(p for _, p in valid_probs)
                confidence = round(float((best_prob / total_valid_prob) * 100), 2) if total_valid_prob > 0 else 90.0

            response = {
                'predicted_disease': predicted_disease,
                'confidence': confidence,
                'risk_level': 'High' if confidence >= 75 else 'Medium' if confidence >= 55 else 'Low'
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            return

        if self.path.startswith('/api/image-disease'):
            try:
                query = urllib.parse.urlparse(self.path).query
                params = urllib.parse.parse_qs(query)
                crop_name = params.get('crop', ['Rice'])[0]

                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length)
                image = Image.open(io.BytesIO(body)).convert('RGB')

                keras_result = predict_image_disease_with_keras(image, crop_name)
                if keras_result is not None:
                    predicted_disease = keras_result['predicted_disease']
                    confidence = keras_result['confidence']
                else:
                    if not os.path.exists(IMAGE_DISEASE_MODEL_PATH):
                        self.send_response(503)
                        self.send_header('Content-Type', 'application/json; charset=utf-8')
                        self.end_headers()
                        self.wfile.write(json.dumps({'error': 'Image disease model not trained yet'}).encode('utf-8'))
                        return

                    image = image.resize((128, 128))
                    pixels = list(image.getdata())
                    brightness = sum(p[0] + p[1] + p[2] for p in pixels) / (len(pixels) * 3 * 255)
                    contrast = sum(abs(p[i] - 128) for p in pixels for i in range(3)) / (len(pixels) * 3 * 127)
                    green_ratio = sum(1 for p in pixels if p[1] > p[0] and p[1] > p[2]) / len(pixels)
                    brown_ratio = sum(1 for p in pixels if p[0] > 90 and p[1] > 60 and p[2] < 80 and p[1] < p[0]) / len(pixels)
                    dark_ratio = sum(1 for p in pixels if sum(p) < 180) / len(pixels)
                    yellow_ratio = sum(1 for p in pixels if p[0] > 180 and p[1] > 150 and p[2] < 80) / len(pixels)
                    edge_density = 0.15 + (contrast * 0.1)

                    bundle = joblib.load(IMAGE_DISEASE_MODEL_PATH)
                    model = bundle['model']
                    crop_encoder = bundle['crop_encoder']
                    disease_encoder = bundle['disease_encoder']
                    feature_columns = bundle['feature_columns']

                    try:
                        encoded_crop = crop_encoder.transform([crop_name])[0]
                    except ValueError:
                        encoded_crop = crop_encoder.transform(['Rice'])[0]

                    payload = {
                        'crop': encoded_crop,
                        'green_ratio': round(green_ratio, 3),
                        'brown_ratio': round(brown_ratio, 3),
                        'dark_ratio': round(dark_ratio, 3),
                        'yellow_ratio': round(yellow_ratio, 3),
                        'edge_density': round(edge_density, 3),
                        'brightness': round(brightness, 3),
                        'contrast': round(contrast, 3),
                    }
                    frame = pd.DataFrame([payload])[feature_columns]
                    probabilities = model.predict_proba(frame)[0]
                    classes = disease_encoder.classes_

                    valid_diseases = CROP_DISEASE_MAPPING.get(crop_name, ['Rice Blast'])
                    valid_probs = [(c, probabilities[idx]) for idx, c in enumerate(classes) if c in valid_diseases]

                    if not valid_probs:
                        predicted_disease = valid_diseases[0]
                        confidence = 90.0
                    else:
                        best_disease, best_prob = max(valid_probs, key=lambda x: x[1])
                        predicted_disease = best_disease
                        total_valid_prob = sum(p for _, p in valid_probs)
                        confidence = round(float((best_prob / total_valid_prob) * 100), 2) if total_valid_prob > 0 else 90.0

                response = {'predicted_disease': predicted_disease, 'confidence': confidence}
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
            except Exception as exc:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(exc)}).encode('utf-8'))
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        # Log errors (4xx, 5xx) but suppress routine request logs
        status = str(args[1]) if len(args) > 1 else ''
        if status.startswith('4') or status.startswith('5'):
            BaseHTTPRequestHandler.log_message(self, format, *args)


if __name__ == '__main__':
    server = OptiCropServer(('0.0.0.0', 8000), OptiCropHandler)
    print('OptiCrop server running at http://127.0.0.1:8000')
    server.serve_forever()
