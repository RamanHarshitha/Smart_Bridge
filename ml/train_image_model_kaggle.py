import argparse
import json
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'image_disease_model_keras.h5'
CLASSES_PATH = ROOT / 'image_disease_model_classes.json'


def make_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    parser = argparse.ArgumentParser(description='Train an image disease model from a Kaggle-style dataset folder.')
    parser.add_argument('--dataset-dir', required=True, help='Path to root image dataset directory')
    parser.add_argument('--epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split ratio')
    args = parser.parse_args()

    data_dir = Path(args.dataset_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f'Dataset directory not found: {data_dir}')

    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=args.validation_split,
        subset='training',
        seed=123,
        image_size=(224, 224),
        batch_size=args.batch_size,
        label_mode='int'
    )
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=args.validation_split,
        subset='validation',
        seed=123,
        image_size=(224, 224),
        batch_size=args.batch_size,
        label_mode='int'
    )

    class_names = train_ds.class_names
    model = make_model(len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), save_best_only=True, monitor='val_loss'),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    model.save(str(MODEL_PATH))
    with open(CLASSES_PATH, 'w', encoding='utf-8') as fh:
        json.dump(class_names, fh, ensure_ascii=False, indent=2)

    print(f'Saved model to {MODEL_PATH}')
    print(f'Saved class labels to {CLASSES_PATH}')
    print('Training history:', history.history)


if __name__ == '__main__':
    main()
