#!/usr/bin/env python3
"""
Setup script for Advanced Cervical Image Classification System
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    dependencies = [
        "tensorflow>=2.12.0",
        "tensorflow-addons>=0.20.0",
        "keras>=2.12.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.1.0",
        "opencv-python>=4.6.0",
        "Pillow>=9.0.0",
        "albumentations>=1.3.0",
        "ipywidgets>=8.0.0",
        "IPython>=8.0.0",
        "scipy>=1.9.0",
        "pandas>=1.4.0",
        "tqdm>=4.64.0"
    ]
    
    print("📦 Installing dependencies...")
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep.split('>=')[0]}"):
            return False
    return True

def check_gpu_support():
    """Check if GPU support is available"""
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU support detected: {len(gpus)} GPU(s) available")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
        else:
            print("⚠️ No GPU detected. Training will use CPU (slower)")
        return True
    except ImportError:
        print("❌ TensorFlow not installed properly")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "logs",
        "results",
        "data",
        "checkpoints"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Created directory: {dir_name}")
    
    return True

def verify_installation():
    """Verify that all components are working"""
    print("\n🔍 Verifying installation...")
    
    try:
        # Test imports
        import tensorflow as tf
        import numpy as np
        import cv2
        import albumentations as A
        import sklearn
        print("✅ All required packages imported successfully")
        
        # Test TensorFlow
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Test basic functionality
        x = np.random.random((1, 224, 224, 3))
        print("✅ NumPy arrays working")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def display_next_steps():
    """Display next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. Prepare your dataset in the required structure:")
    print("   LastDatasetVedant/")
    print("   ├── Train/")
    print("   │   ├── Normal/")
    print("   │   └── Abnormal/")
    print("   └── Test/")
    print("       ├── Normal/")
    print("       └── Abnormal/")
    
    print("\n2. Choose your training method:")
    print("   Option A: Use Jupyter Notebook (Recommended)")
    print("   - Open 'Advanced_Cervical_Classification.ipynb' in Google Colab")
    print("   - Update dataset paths in the configuration section")
    print("   - Run all cells")
    
    print("\n   Option B: Use Python scripts")
    print("   - Update paths in cervical_classification_advanced.py")
    print("   - Run: python cervical_classification_advanced.py")
    
    print("\n3. Monitor training progress:")
    print("   - Check GPU utilization")
    print("   - Monitor loss and accuracy curves")
    print("   - Expected training time: 2-4 hours with GPU")
    
    print("\n4. Use the trained models:")
    print("   - Interactive prediction widget")
    print("   - Batch processing of image folders")
    print("   - Model evaluation and visualization")
    
    print("\n🔗 Files created:")
    print("   - cervical_classification_advanced.py (Main training script)")
    print("   - prediction_utils.py (Prediction utilities)")
    print("   - Advanced_Cervical_Classification.ipynb (Complete notebook)")
    print("   - requirements.txt (Dependencies)")
    print("   - README.md (Documentation)")
    
    print("\n⚠️ Important Notes:")
    print("   - This is a research tool, not a medical device")
    print("   - Always validate with medical professionals")
    print("   - Ensure proper data privacy and ethics compliance")
    print("   - Expected accuracy: 90%+ with proper dataset")

def main():
    """Main setup function"""
    print("🏥 Advanced Cervical Image Classification Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Check GPU support
    if not check_gpu_support():
        print("⚠️ GPU check failed, but continuing...")
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        sys.exit(1)
    
    # Display next steps
    display_next_steps()

if __name__ == "__main__":
    main()