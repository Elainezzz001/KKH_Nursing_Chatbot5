#!/usr/bin/env python3
"""
Test script for KKH Nursing Chatbot
This script verifies that all dependencies are installed and basic functionality works.
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'streamlit',
        'requests', 
        'PyPDF2',
        'faiss',
        'numpy',
        'sentence_transformers',
        'torch',
        'transformers'
    ]
    
    print("Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def test_lm_studio_connection():
    """Test connection to LM Studio"""
    import requests
    from config import LM_STUDIO_HOST, LM_STUDIO_PORT
    
    url = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1/models"
    
    print(f"\nTesting LM Studio connection at {url}...")
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print("✅ LM Studio is running and accessible")
            return True
        else:
            print(f"❌ LM Studio returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to LM Studio")
        print("   Please ensure LM Studio is running and the server is started")
        return False
    except Exception as e:
        print(f"❌ Error connecting to LM Studio: {e}")
        return False

def test_pdf_file():
    """Test if PDF file exists"""
    import os
    from config import PDF_PATH
    
    print(f"\nTesting PDF file at {PDF_PATH}...")
    
    if os.path.exists(PDF_PATH):
        print("✅ PDF file found")
        
        # Test if file is readable
        try:
            with open(PDF_PATH, 'rb') as f:
                f.read(100)  # Try to read first 100 bytes
            print("✅ PDF file is readable")
            return True
        except Exception as e:
            print(f"❌ Cannot read PDF file: {e}")
            return False
    else:
        print(f"❌ PDF file not found at {PDF_PATH}")
        return False

def test_embedding_model():
    """Test if embedding model can be loaded"""
    print("\nTesting embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        from config import EMBEDDING_MODEL
        
        print(f"Loading {EMBEDDING_MODEL}...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Test encoding
        test_text = "This is a test sentence."
        embedding = model.encode([test_text])
        
        print(f"✅ Embedding model loaded successfully")
        print(f"   Embedding dimension: {embedding.shape[1]}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        print("   This may take time on first run to download the model")
        return False

def main():
    """Run all tests"""
    print("🏥 KKH Nursing Chatbot - System Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("LM Studio Connection", test_lm_studio_connection),
        ("PDF File", test_pdf_file),
        ("Embedding Model", test_embedding_model)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 All tests passed! The chatbot should work correctly.")
        print("\nTo start the application, run:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues before running the chatbot.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Start LM Studio and load the model")
        print("- Ensure PDF file is in the correct location")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
