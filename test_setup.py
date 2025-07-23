"""
Test script for KKH Nursing Chatbot
Run this to verify all components are working correctly
"""

import os
import sys
import json
import requests
from pathlib import Path

def test_file_structure():
    """Test if all required files exist"""
    print("🔍 Testing file structure...")
    
    required_files = [
        "app.py",
        "requirements.txt",
        "quiz_data.json",
        "data/KKH Information file.pdf",
        "logo/photo_2025-06-16_15-57-21.jpg"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_dependencies():
    """Test if all required Python packages are installed"""
    print("\n🔍 Testing dependencies...")
    
    required_packages = [
        "streamlit",
        "sentence_transformers", 
        "PyPDF2",
        "requests",
        "numpy",
        "sklearn",
        "pandas"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies satisfied")
        return True

def test_quiz_data():
    """Test if quiz data is valid JSON"""
    print("\n🔍 Testing quiz data...")
    
    try:
        with open("quiz_data.json", "r") as f:
            quiz_data = json.load(f)
        
        if not isinstance(quiz_data, list):
            print("❌ Quiz data should be a list")
            return False
        
        if len(quiz_data) < 15:
            print(f"⚠️  Only {len(quiz_data)} questions (recommended: 15+)")
        
        # Test first question structure
        if quiz_data:
            q = quiz_data[0]
            required_keys = ["question", "options", "correct_answer", "explanation"]
            
            for key in required_keys:
                if key not in q:
                    print(f"❌ Missing key '{key}' in quiz question")
                    return False
            
            if not isinstance(q["options"], dict) or len(q["options"]) != 4:
                print("❌ Each question should have exactly 4 options")
                return False
        
        print(f"✅ Quiz data valid ({len(quiz_data)} questions)")
        return True
        
    except json.JSONDecodeError:
        print("❌ Invalid JSON in quiz_data.json")
        return False
    except FileNotFoundError:
        print("❌ quiz_data.json not found")
        return False

def test_pdf_access():
    """Test if PDF file can be accessed"""
    print("\n🔍 Testing PDF access...")
    
    pdf_path = "data/KKH Information file.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return False
    
    try:
        file_size = os.path.getsize(pdf_path)
        if file_size == 0:
            print("❌ PDF file is empty")
            return False
        
        print(f"✅ PDF accessible ({file_size / 1024 / 1024:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"❌ Error accessing PDF: {e}")
        return False

def test_lm_studio_connection():
    """Test connection to LM Studio"""
    print("\n🔍 Testing LM Studio connection...")
    
    lm_studio_url = "http://192.168.75.1:1234/v1/models"
    
    try:
        response = requests.get(lm_studio_url, timeout=5)
        if response.status_code == 200:
            print("✅ LM Studio connected and responding")
            return True
        else:
            print(f"⚠️  LM Studio responded with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("⚠️  Cannot connect to LM Studio (this is OK for deployment)")
        print("   Make sure LM Studio is running on http://192.168.75.1:1234")
        return False
    except Exception as e:
        print(f"⚠️  LM Studio connection error: {e}")
        return False

def test_embedding_model():
    """Test if embedding model can be loaded"""
    print("\n🔍 Testing embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("Loading embedding model (this may take a moment)...")
        model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
        
        # Test encoding
        test_text = "This is a test sentence"
        embedding = model.encode([test_text])
        
        if embedding is not None and len(embedding) > 0:
            print(f"✅ Embedding model loaded successfully (dimension: {len(embedding[0])})")
            return True
        else:
            print("❌ Embedding model failed to encode text")
            return False
            
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        return False

def test_streamlit_app():
    """Test if Streamlit app can be imported"""
    print("\n🔍 Testing Streamlit app import...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, ".")
        
        # Try to import the app (this will test syntax)
        import app
        print("✅ Streamlit app imports successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error importing app: {e}")
        return False

def main():
    """Run all tests"""
    print("🏥 KKH Nursing Chatbot - System Test\n")
    
    tests = [
        ("File Structure", test_file_structure),
        ("Dependencies", test_dependencies),
        ("Quiz Data", test_quiz_data),
        ("PDF Access", test_pdf_access),
        ("LM Studio Connection", test_lm_studio_connection),
        ("Embedding Model", test_embedding_model),
        ("Streamlit App", test_streamlit_app)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your KKH Nursing Chatbot is ready to use.")
        print("Run: streamlit run app.py")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please address the issues above.")
    
    return passed == len(results)

if __name__ == "__main__":
    main()
