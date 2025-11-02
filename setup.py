





"""
Setup script for Emotion Detection Project
Run this to verify your setup before deployment
"""

import os
import sys

def check_file(filename):
    """Check if a file exists"""
    exists = os.path.exists(filename)
    status = "✅" if exists else "❌"
    print(f"{status} {filename}")
    return exists

def check_requirements():
    """Check if all required files are present"""
    print("\n📋 Checking Project Files...")
    print("-" * 50)
    
    required_files = [
        'app.py',
        'model.py',
        'requirements.txt',
        'README.md',
        'deployment_url.txt',
        '.gitignore'
    ]
    
    optional_files = [
        'emotion_model.h5',
        'emotion_training_colab.ipynb'
    ]
    
    all_present = True
    
    print("\nRequired Files:")
    for file in required_files:
        if not check_file(file):
            all_present = False
    
    print("\nOptional Files:")
    for file in optional_files:
        check_file(file)
    
    return all_present

def check_python_version():
    """Check Python version"""
    print("\n🐍 Checking Python Version...")
    print("-" * 50)
    version = sys.version_info
    print(f"Current version: Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8+ is required")
        return False

def test_imports():
    """Test if required packages can be imported"""
    print("\n📦 Testing Package Imports...")
    print("-" * 50)
    
    packages = {
        'streamlit': 'Streamlit',
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'PIL': 'Pillow'
    }
    
    all_imported = True
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Not installed")
            all_imported = False
    
    return all_imported

def create_directory_structure():
    """Create necessary directories"""
    print("\n📁 Directory Structure...")
    print("-" * 50)
    print("Current structure is flat (all files in root)")
    print("✅ No additional directories needed")

def display_next_steps():
    """Display next steps for deployment"""
    print("\n🚀 Next Steps for Deployment...")
    print("-" * 50)
    print("""
1. TRAIN YOUR MODEL (Google Colab):
   - Upload emotion_training_colab.ipynb to Google Colab
   - Follow the notebook instructions
   - Download emotion_model.h5
   - Place it in your project root

2. TEST LOCALLY:
   - Run: streamlit run app.py
   - Test with sample images
   - Verify everything works

3. PUSH TO GITHUB:
   - git init
   - git add .
   - git commit -m "Initial commit"
   - Create GitHub repository
   - git remote add origin <your-repo-url>
   - git push -u origin main

4. DEPLOY TO RENDER:
   - Sign up at render.com
   - Create new Web Service
   - Connect GitHub repository
   - Build Command: pip install -r requirements.txt
   - Start Command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   - Deploy!

5. UPDATE deployment_url.txt:
   - Add your Render URL
   - Add your GitHub URL
   - Include in final submission

6. CREATE ZIP FILE:
   - Name: surname_matric.zip
   - Include all project files
   - Submit!
    """)

def main():
    """Main setup verification"""
    print("=" * 50)
    print("🎭 EMOTION DETECTION PROJECT SETUP")
    print("=" * 50)
    
    files_ok = check_requirements()
    python_ok = check_python_version()
    
    if files_ok and python_ok:
        imports_ok = test_imports()
        
        if not imports_ok:
            print("\n⚠️  Some packages are missing. Install them with:")
            print("    pip install -r requirements.txt")
    
    create_directory_structure()
    display_next_steps()
    
    print("\n" + "=" * 50)
    if files_ok:
        print("✅ Setup verification complete!")
    else:
        print("⚠️  Some files are missing. Check the output above.")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
