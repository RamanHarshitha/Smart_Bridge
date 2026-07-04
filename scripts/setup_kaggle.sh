#!/bin/bash
# Quick setup script to download Kaggle dataset and retrain crop model

set -e

echo "================================"
echo "OptiCrop - Kaggle Dataset Setup"
echo "================================"
echo ""

WORK_DIR="/home/rguktrkvalley/Desktop/Smart_Bridge"
CSV_FILE="$WORK_DIR/Crop_recommendation.csv"
ML_DIR="$WORK_DIR/ml"

# Check if CSV already exists
if [ -f "$CSV_FILE" ]; then
    echo "✓ Crop_recommendation.csv already exists"
else
    echo "⚠ Crop_recommendation.csv not found"
    echo ""
    echo "To download the Kaggle dataset, run:"
    echo "  1. pip install kaggle"
    echo "  2. Go to https://www.kaggle.com/settings/account and download kaggle.json"
    echo "  3. Place kaggle.json in ~/.kaggle/"
    echo "  4. Then run:"
    echo "     kaggle datasets download -d atharvaingle/crop-recommendation-dataset"
    echo "     unzip -o crop-recommendation-dataset.zip"
    echo ""
    exit 1
fi

echo ""
echo "Verifying dataset..."
python3 -c "
import pandas as pd
df = pd.read_csv('$CSV_FILE')
print(f'✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features')
print(f'✓ Columns: {list(df.columns)}')
print(f'✓ Crops: {df[\"label\"].nunique()} types')
print(f'✓ Crop examples: {list(df[\"label\"].unique()[:5])}')
"

echo ""
echo "Starting model training..."
cd "$ML_DIR"
python3 train_model.py

echo ""
echo "✓ Model training complete!"
echo ""
echo "To use the new model, restart the OptiCrop server:"
echo "  cd $WORK_DIR && python server.py"
