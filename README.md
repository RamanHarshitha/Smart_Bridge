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

## Dataset setup for full functionality

The app can run with the bundled fallback data, but for richer results you can add Kaggle datasets locally.

### 1) Crop recommendation dataset

Download a Kaggle crop recommendation dataset that contains columns similar to:

- nitrogen (N)
- phosphorous (P)
- potassium (K)
- temperature
- humidity
- ph
- rainfall
- label

Then place the downloaded CSV in:

```text
data/Crop_recommendation.csv
```

Example workflow:

```bash
pip install kaggle
mkdir -p ~/.kaggle
# place your kaggle.json file in ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d <your-kaggle-dataset-slug> -p data/
unzip data/*.zip -d data/
# rename the resulting CSV to the expected file name if needed
mv data/<downloaded-file>.csv data/Crop_recommendation.csv
```

After that, retrain the crop model:

```bash
python3 ml/train_model.py
```

### 2) Symptom-based disease detection dataset

The current symptom-based disease module uses the disease records bundled in [data/opticrop-data.json](data/opticrop-data.json).

If you want to retrain or expand it with a Kaggle disease dataset, prepare a CSV or JSON file with fields such as:

- crop
- temperature
- humidity
- ph
- disease
- symptoms

Then place it in the expected data location and update the training script if needed.

Example idea:

```bash
# download a Kaggle disease metadata or symptom dataset
kaggle datasets download -d <your-disease-dataset-slug> -p data/
unzip data/*.zip -d data/
```

### 3) Image-based disease detection dataset

For image-based disease detection, download a Kaggle plant disease image dataset such as:

- New Plant Diseases Dataset (Augmented)
- Plant Village / plant disease image datasets

Extract the dataset into a folder such as:

```text
data/datasets/plant_disease/
```

Then train the image model with:

```bash
python3 ml/train_image_model_kaggle.py --dataset-dir data/datasets/plant_disease
```

If you use a different folder structure, adjust the path accordingly.

## Train the crop model

If the Kaggle CSV is available in the data folder, you can retrain the crop model with:

```bash
python3 ml/train_model.py
```

## Notes

- The crop recommendation pipeline prefers the Kaggle dataset when present in [data/Crop_recommendation.csv](data/Crop_recommendation.csv).
- If that file is missing, the app falls back to the bundled JSON dataset.
- The trained model files are stored in [models](models).
- The image disease module is optional and works best when the full Kaggle image dataset is present locally.

## Push the updated README to GitHub

After saving the changes, run:

```bash
git add README.md
git commit -m "Update README with dataset setup instructions"
git push origin master
```

If your repository uses a different branch name, replace `master` with your branch name.
