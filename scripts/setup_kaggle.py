#!/usr/bin/env python3
"""
OptiCrop - Kaggle Dataset Setup Script
Downloads and retrains the crop recommendation model with Kaggle data
"""

import os
import sys
import subprocess
import pandas as pd

WORK_DIR = "/home/rguktrkvalley/Desktop/Smart_Bridge"
CSV_FILE = os.path.join(WORK_DIR, "Crop_recommendation.csv")
ML_DIR = os.path.join(WORK_DIR, "ml")

def print_header():
    print("\n" + "="*50)
    print("OptiCrop - Kaggle Dataset Setup")
    print("="*50 + "\n")

def check_csv_exists():
    if os.path.exists(CSV_FILE):
        print(f"✓ Found: {CSV_FILE}")
        return True
    return False

def download_instructions():
    print("⚠ Crop_recommendation.csv not found\n")
    print("To download the Kaggle dataset, follow these steps:\n")
    print("  1. Install Kaggle CLI:")
    print("     pip install kaggle\n")
    print("  2. Get API credentials:")
    print("     - Go to https://www.kaggle.com/settings/account")
    print("     - Click 'Create New API Token'")
    print("     - Save kaggle.json to ~/.kaggle/ (or C:\\Users\\<name>\\.kaggle\\ on Windows)\n")
    print("  3. Download dataset:")
    print(f"     cd {WORK_DIR}")
    print("     kaggle datasets download -d atharvaingle/crop-recommendation-dataset")
    print("     unzip crop-recommendation-dataset.zip\n")
    print("  4. Run this script again")
    return False

def verify_dataset():
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        print(f"✓ Columns: {list(df.columns)}")
        print(f"✓ Crops: {df['label'].nunique()} types")
        print(f"✓ Sample crops: {', '.join(list(df['label'].unique()[:5]))}")
        
        # Verify required columns
        required_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"✗ Missing columns: {missing}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error reading dataset: {e}")
        return False

def train_model():
    print("\n" + "-"*50)
    print("Training crop recommendation model...")
    print("-"*50 + "\n")
    
    try:
        os.chdir(ML_DIR)
        result = subprocess.run([sys.executable, "train_model.py"], check=True)
        print("\n✓ Model training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False

def main():
    print_header()
    
    # Check CSV
    if not check_csv_exists():
        if not download_instructions():
            print("\n✗ Setup incomplete. Please download the dataset first.")
            return False
        return False
    
    # Verify dataset
    print("\nVerifying dataset...")
    if not verify_dataset():
        print("\n✗ Dataset verification failed.")
        return False
    
    # Train model
    if not train_model():
        print("\n✗ Model training failed.")
        return False
    
    # Success
    print("\n" + "="*50)
    print("✓ Setup Complete!")
    print("="*50)
    print("\nNext steps:")
    print(f"  1. Restart OptiCrop server:")
    print(f"     cd {WORK_DIR}")
    print(f"     python server.py")
    print(f"\n  2. Visit: http://127.0.0.1:8000")
    print(f"\n  3. Try crop recommendation with new Kaggle-based model\n")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
