# Smart Bridge

Smart Bridge is a lightweight agricultural decision-support web app that combines crop recommendation, crop suitability analysis, fertilizer guidance, and plant disease assistance in one place.

## What this project includes

- Crop recommendation using a trained machine learning model
- Crop suitability and environmental assessment
- Fertilizer recommendations based on crop and soil context
- Disease prediction for common crop problems
- A simple local web interface served by a Python HTTP server

## Project structure

- data/: datasets and raw input files
- models/: trained ML model artifacts
- web/: frontend HTML, CSS, and JavaScript files
- ml/: training and inference scripts for machine learning models
- docs/: setup notes and project documentation
- scripts/: convenience scripts for setup and training
- server.py: local web server entry point

## Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

From the project root:

```bash
python3 server.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Train the crop model

If the Kaggle CSV is available in the data folder, you can retrain the crop model with:

```bash
python3 ml/train_model.py
```

## Notes

- The crop recommendation pipeline prefers the Kaggle dataset when present in [data/Crop_recommendation.csv](data/Crop_recommendation.csv).
- If that file is missing, the app falls back to the bundled JSON dataset.
- The trained model files are stored in [models](models).
