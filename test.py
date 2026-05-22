"""
AI Image Detector - Testing Script
This script evaluates the trained model on the test dataset.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

np.random.seed(42)
tf.random.set_seed(42)

CONFIG = {
    'dataset_path': 'dataset',
    'model_path': 'outputs/best_model.keras',
    'class_indices_path': 'outputs/class_indices.json',
    'img_size': (224, 224),
    'batch_size': 32,
}

def load_model_and_classes():
    print("📂 Loading model and class indices...")
    
    if not os.path.exists(CONFIG['model_path']):
        print(f"❌ Error: Model not found at '{CONFIG['model_path']}'")
        print("   Please train the model first using 'python train.py'")
        return None, None
    
    model = keras.models.load_model(CONFIG['model_path'])
    print(f"   ✓ Model loaded from {CONFIG['model_path']}")
    
    if not os.path.exists(CONFIG['class_indices_path']):
        print(f"❌ Error: Class indices not found at '{CONFIG['class_indices_path']}'")
        return None, None
    
    with open(CONFIG['class_indices_path'], 'r') as f:
        class_indices = json.load(f)
    
    class_names = {v: k for k, v in class_indices.items()}
    print(f"   ✓ Classes: {class_names}")
    
    return model, class_names

def create_test_generator():
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(CONFIG['dataset_path'], 'test'),
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        shuffle=False
    )
    
    return test_generator

def plot_confusion_matrix(cm, class_names, save_path='outputs/confusion_matrix.png'):
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Confusion matrix saved to {save_path}")

def plot_sample_predictions(model, test_generator, class_names, num_samples=9):
    images, labels = next(test_generator)
    predictions = model.predict(images[:num_samples])
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        true_label = class_names[np.argmax(labels[i])]
        pred_label = class_names[np.argmax(predictions[i])]
        confidence = np.max(predictions[i]) * 100
        
        color = 'green' if true_label == pred_label else 'red'
        
        axes[i].imshow(images[i])
        axes[i].axis('off')
        axes[i].set_title(
            f'True: {true_label}\nPredicted: {pred_label}\nConfidence: {confidence:.1f}%',
            color=color,
            fontweight='bold'
        )
    
    plt.tight_layout()
    plt.savefig('outputs/sample_predictions.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Sample predictions saved to outputs/sample_predictions.png")

def evaluate_model(model, test_generator, class_names):
    print("\n🎯 Evaluating model on test set...")
    
    print("   • Generating predictions...")
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    true_classes = test_generator.classes
    
    print("\n📊 Performance Metrics:")
    print("=" * 60)
    
    accuracy = accuracy_score(true_classes, predicted_classes)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    precision = precision_score(true_classes, predicted_classes, average='weighted')
    recall = recall_score(true_classes, predicted_classes, average='weighted')
    f1 = f1_score(true_classes, predicted_classes, average='weighted')
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 60)
    
    print("\n📋 Detailed Classification Report:")
    print("-" * 60)
    class_names_list = [class_names[i] for i in sorted(class_names.keys())]
    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names_list,
        digits=4
    )
    print(report)
    
    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(true_classes, predicted_classes)
    print(cm)
    
    print("\n📈 Generating visualizations...")
    plot_confusion_matrix(cm, class_names_list)
    plot_sample_predictions(model, test_generator, class_names)
    
    with open('outputs/test_metrics.txt', 'w') as f:
        f.write("AI IMAGE DETECTOR - TEST RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write("-" * 60 + "\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    
    print(f"   ✓ Test metrics saved to outputs/test_metrics.txt")
    
    return accuracy, precision, recall, f1

def main():
    print("=" * 80)
    print("🧪 AI IMAGE DETECTOR - TESTING PIPELINE")
    print("=" * 80)
    
    test_path = os.path.join(CONFIG['dataset_path'], 'test')
    if not os.path.exists(test_path):
        print(f"❌ Error: Test dataset not found at '{test_path}'")
        return
    
    os.makedirs('outputs', exist_ok=True)
    
    model, class_names = load_model_and_classes()
    if model is None:
        return
    
    print("\n📂 Loading test data...")
    test_generator = create_test_generator()
    print(f"   ✓ Test samples: {test_generator.samples}")
    
    accuracy, precision, recall, f1 = evaluate_model(model, test_generator, class_names)
    
    print("\n" + "=" * 80)
    print("✅ TESTING COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()