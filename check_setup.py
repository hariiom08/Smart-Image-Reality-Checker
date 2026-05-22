"""
Environment Setup Checker
Run this script to verify your environment is properly configured.
"""

import sys
import os

def check_python_version():
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (Compatible)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False

def check_packages():
    print("\n📦 Checking required packages...")
    
    required_packages = {
        'tensorflow': 'TensorFlow',
        'keras': 'Keras',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn',
        'streamlit': 'Streamlit',
    }
    
    all_installed = True
    
    for module, name in required_packages.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} (Not installed)")
            all_installed = False
    
    return all_installed

def check_tensorflow_gpu():
    print("\n🖥️  Checking GPU availability...")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   ✅ {len(gpus)} GPU(s) detected")
            return True
        else:
            print("   ℹ️  No GPU detected (CPU will be used)")
            return True
    except Exception as e:
        print(f"   ⚠️  Error checking GPU: {str(e)}")
        return True

def check_dataset_structure():
    print("\n📁 Checking dataset structure...")
    
    required_paths = [
        'dataset/train/ai',
        'dataset/train/real',
        'dataset/test/ai',
        'dataset/test/real'
    ]
    
    all_exist = True
    
    for path in required_paths:
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"   ✅ {path} ({count} images)")
        else:
            print(f"   ❌ {path} (Not found)")
            all_exist = False
    
    if not all_exist:
        print("\n   ℹ️  To create the dataset structure:")
        print("      mkdir -p dataset/train/ai dataset/train/real dataset/test/ai dataset/test/real")
    
    return all_exist

def check_file_structure():
    print("\n📄 Checking project files...")
    
    required_files = [
        'train.py',
        'test.py',
        'app.py',
        'requirements.txt',
    ]
    
    all_exist = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (Missing)")
            all_exist = False
    
    return all_exist

def main():
    print("=" * 80)
    print("🔍 AI IMAGE DETECTOR - ENVIRONMENT CHECK")
    print("=" * 80)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Required Packages", check_packages()),
        ("GPU Availability", check_tensorflow_gpu()),
        ("Project Files", check_file_structure()),
        ("Dataset Structure", check_dataset_structure()),
    ]
    
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    
    for name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(passed for _, passed in checks)
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to start training.")
        print("\n🚀 Next steps:")
        print("   1. Make sure your dataset is populated with images")
        print("   2. Run: python train.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\n💡 Common fixes:")
        print("   • Missing packages: pip install -r requirements.txt")
        print("   • Missing files: Download all files from this chat")
        print("   • Missing dataset: Create folders and add images")

if __name__ == "__main__":
    main()


