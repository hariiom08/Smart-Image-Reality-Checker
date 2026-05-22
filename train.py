"""
AI Image Detector - Training Script
This script trains a deep learning model to classify images as AI-generated or real.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
CONFIG = {
    'dataset_path': 'dataset',
    'img_size': (224, 224),
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.0001,
    'validation_split': 0.2,
}

def create_model(num_classes=2):
    base_model = MobileNetV2(
        input_shape=(*CONFIG['img_size'], 3),
        include_top=False,
        weights='imagenet'
    )
    # Unfreeze the last 50 layers for fine-tuning
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    for layer in base_model.layers[-50:]:
        layer.trainable = True
    
    model = keras.Sequential([
        layers.Input(shape=(*CONFIG['img_size'], 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_data_generators():
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=CONFIG['validation_split']
    )
    
    validation_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=CONFIG['validation_split']
    )
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(CONFIG['dataset_path'], 'train'),
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(CONFIG['dataset_path'], 'train'),
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

def plot_training_history(history, save_path='training_history.png'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss Over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Training history plot saved to {save_path}")

def main():
    print("=" * 80)
    print("🚀 AI IMAGE DETECTOR - TRAINING PIPELINE")
    print("=" * 80)
    
    if not os.path.exists(CONFIG['dataset_path']):
        print(f"❌ Error: Dataset not found at '{CONFIG['dataset_path']}'")
        return
    
    os.makedirs('outputs', exist_ok=True)
    
    print("\n📊 STEP 1: Loading and preparing data...")
    train_generator, validation_generator = create_data_generators()
    
    print(f"   ✓ Training samples: {train_generator.samples}")
    print(f"   ✓ Validation samples: {validation_generator.samples}")
    
    print("\n🏗️  STEP 2: Building model...")
    model = create_model(num_classes=len(train_generator.class_indices))
    model.summary()
    
    print("\n⚙️  STEP 3: Setting up callbacks...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint('outputs/best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    print("\n🎯 STEP 4: Training the model...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n📈 STEP 5: Plotting training history...")
    plot_training_history(history, 'outputs/training_history.png')
    
    print("\n💾 STEP 6: Saving final model...")
    model.save('outputs/final_model.keras')
    
    import json
    with open('outputs/class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)
    
    print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()


