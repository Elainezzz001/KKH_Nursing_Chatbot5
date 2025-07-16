#!/usr/bin/env python3
"""
Test script for KKH Nursing Chatbot
Verifies that all dependencies are installed and can be imported.
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Streamlit: {e}")
        return False
    
    try:
        import requests
        print("✅ Requests imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Requests: {e}")
        return False
    
    try:
        import PyPDF2
        print("✅ PyPDF2 imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PyPDF2: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✅ Sentence Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Sentence Transformers: {e}")
        return False
    
    try:
        import faiss
        print("✅ FAISS imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import FAISS: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import NumPy: {e}")
        return False
    
    return True

def test_file_structure():
    """Test required files and directories"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "app.py",
        "requirements.txt",
        "Dockerfile",
        "fly.toml"
    ]
    
    required_dirs = [
        "data",
        "logo"
    ]
    
    all_good = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_good = False
    
    for dir in required_dirs:
        if os.path.exists(dir):
            print(f"✅ {dir}/ directory exists")
        else:
            print(f"❌ {dir}/ directory missing")
            all_good = False
    
    # Check for PDF file
    pdf_path = "data/KKH Information file.pdf"
    if os.path.exists(pdf_path):
        print(f"✅ KKH Information file.pdf exists")
    else:
        print(f"⚠️  KKH Information file.pdf missing (required for full functionality)")
    
    return all_good

def test_lm_studio_connection():
    """Test connection to LM Studio"""
    print("\n🤖 Testing LM Studio connection...")
    
    try:
        import requests
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ LM Studio is running and accessible")
            return True
        else:
            print(f"⚠️  LM Studio returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("⚠️  LM Studio is not running or not accessible at http://localhost:1234")
        print("   Please start LM Studio and load the openhermes-2.5-mistral-7b model")
        return False
    except Exception as e:
        print(f"❌ Error testing LM Studio connection: {e}")
        return False

def main():
    """Run all tests"""
    print("🏥 KKH Nursing Chatbot - System Test")
    print("===================================")
    
    # Test imports
    imports_ok = test_imports()
    
    # Test file structure
    files_ok = test_file_structure()
    
    # Test LM Studio connection
    lm_studio_ok = test_lm_studio_connection()
    
    print("\n📊 Test Summary:")
    print("================")
    
    if imports_ok:
        print("✅ All Python dependencies are installed")
    else:
        print("❌ Some Python dependencies are missing")
    
    if files_ok:
        print("✅ All required files are present")
    else:
        print("❌ Some required files are missing")
    
    if lm_studio_ok:
        print("✅ LM Studio connection successful")
    else:
        print("⚠️  LM Studio not accessible (optional for testing)")
    
    print(f"\n🐍 Python version: {sys.version}")
    print(f"📂 Working directory: {os.getcwd()}")
    
    if imports_ok and files_ok:
        print("\n🚀 Ready to run! Execute: streamlit run app.py")
    else:
        print("\n🔧 Please fix the issues above before running the application")

if __name__ == "__main__":
    main()
