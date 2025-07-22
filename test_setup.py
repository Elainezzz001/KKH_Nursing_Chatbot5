#!/usr/bin/env python3
"""
Test script for KKH Nursing Chatbot setup
Run this script to verify all components are working correctly
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'sentence_transformers',
        'faiss',
        'fitz',  # PyMuPDF
        'requests',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed!")
    return True

def check_files():
    """Check if required files exist"""
    print("\n📁 Checking required files...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'data/KKH Information file.pdf'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files present!")
    return True

def check_environment():
    """Check environment variables"""
    print("\n🔧 Checking environment variables...")
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        print("  ✅ OPENROUTER_API_KEY is set")
        print(f"  🔑 Key preview: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else ''}")
    else:
        print("  ⚠️  OPENROUTER_API_KEY not set")
        print("  💡 Set it with: export OPENROUTER_API_KEY=your_key_here")
        print("  💡 Or create a .env file with: OPENROUTER_API_KEY=your_key_here")
    
    return True

def run_basic_tests():
    """Run basic functionality tests"""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test PDF processing
        print("  📄 Testing PDF processing...")
        import fitz
        doc = fitz.open("data/KKH Information file.pdf")
        page_count = len(doc)
        doc.close()
        print(f"     ✅ PDF loaded successfully ({page_count} pages)")
        
        # Test embedding model (just check if it can be imported)
        print("  🔤 Testing embedding model import...")
        from sentence_transformers import SentenceTransformer
        print("     ✅ SentenceTransformer imported successfully")
        
        # Test FAISS
        print("  🔍 Testing FAISS...")
        import faiss
        print("     ✅ FAISS imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🏥 KKH Nursing Chatbot Setup Test")
    print("=" * 40)
    
    all_good = True
    
    all_good &= check_dependencies()
    all_good &= check_files()
    all_good &= check_environment()
    all_good &= run_basic_tests()
    
    print("\n" + "=" * 40)
    
    if all_good:
        print("🎉 All tests passed! You're ready to run the chatbot.")
        print("\n🚀 To start the application, run:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Some issues found. Please fix them before running the chatbot.")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
