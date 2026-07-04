import argparse
import os
from pathlib import Path

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}


def scan_dataset(dataset_dir: Path):
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f'Dataset directory not found: {dataset_dir}')

    classes = {}
    for item in sorted(dataset_dir.iterdir()):
        if item.is_dir():
            count = 0
            for root, _, files in os.walk(item):
                for file in files:
                    if Path(file).suffix.lower() in VALID_EXTENSIONS:
                        count += 1
            if count > 0:
                classes[item.name] = count

    return classes


def main():
    parser = argparse.ArgumentParser(description='Validate a Kaggle-style image dataset folder.')
    parser.add_argument('--dataset-dir', default='../datasets/plant_disease', help='Path to the dataset directory')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    try:
        classes = scan_dataset(dataset_dir)
    except FileNotFoundError as exc:
        print(exc)
        return

    if not classes:
        print('No image classes found. Ensure the dataset directory contains subfolders for each disease label.')
        print(f'Expected class folders under: {dataset_dir}')
        return

    total = sum(classes.values())
    print(f'Dataset root: {dataset_dir}')
    print(f'Found {len(classes)} classes with {total} images total.')
    print('\nClass counts:')
    for cls, count in classes.items():
        print(f'  - {cls}: {count}')

    print('\nDataset structure looks valid for training. Next, run:')
    print('  python3 train_image_model_kaggle.py --dataset-dir', dataset_dir)


if __name__ == '__main__':
    main()
