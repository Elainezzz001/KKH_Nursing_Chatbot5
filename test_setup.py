#!/usr/bin/env python3
"""
Test script for KKH Nursing Chatbot
Run this to validate basic functionality before deployment
"""

import os
import sys
import subprocess
import importlib
import fitz  # PyMuPDF

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Requires Python 3.8+")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'faiss',
        'sentence_transformers',
        'fitz',  # PyMuPDF
        'requests',
        'numpy',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'faiss':
                importlib.import_module('faiss')
            elif package == 'fitz':
                importlib.import_module('fitz')
            else:
                importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📝 To install missing packages, run:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_pdf_file():
    """Check if the PDF file exists and is readable"""
    print("\n📄 Checking PDF file...")
    
    pdf_path = "data/KKH Information file.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found at {pdf_path}")
        return False
    
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        print(f"✅ PDF file found with {page_count} pages")
        return True
    except Exception as e:
        print(f"❌ Error reading PDF file: {e}")
        return False

def check_streamlit():
    """Check if Streamlit can run"""
    print("\n🎯 Testing Streamlit...")
    
    try:
        result = subprocess.run(['streamlit', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Streamlit is working: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Streamlit error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Streamlit command timed out")
        return False
    except FileNotFoundError:
        print("❌ Streamlit command not found")
        return False

def check_secrets():
    """Check if secrets file exists"""
    print("\n🔐 Checking secrets configuration...")
    
    secrets_path = ".streamlit/secrets.toml"
    
    if os.path.exists(secrets_path):
        print("✅ Secrets file found")
        with open(secrets_path, 'r') as f:
            content = f.read()
            if 'OPENROUTER_API_KEY' in content and 'your-openrouter-api-key-here' not in content:
                print("✅ API key appears to be configured")
                return True
            else:
                print("⚠️  API key needs to be configured in .streamlit/secrets.toml")
                return False
    else:
        print("⚠️  Secrets file not found. Create .streamlit/secrets.toml with your API key")
        return False

def main():
    """Run all tests"""
    print("🔬 KKH Nursing Chatbot - Pre-deployment Tests")
    print("=" * 50)
    
    tests = [
        check_python_version,
        check_dependencies,
        check_pdf_file,
        check_streamlit,
        check_secrets
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for deployment.")
        print("\n🚀 To run locally:")
        print("   streamlit run app.py")
        print("\n☁️  To deploy to Fly.io:")
        print("   ./deploy.sh (Linux/Mac) or deploy.bat (Windows)")
    else:
        print("❌ Some tests failed. Please fix the issues before deployment.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
