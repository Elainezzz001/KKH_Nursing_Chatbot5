@echo off
REM Local development script for KKH Nursing Chatbot (Windows)

echo 🏥 KKH Nursing Chatbot - Local Development
echo ========================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if virtual environment exists, create if not
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Check if secrets file exists
if not exist ".streamlit\secrets.toml" (
    echo ⚠️  Secrets file not found. Creating from template...
    copy "secrets.toml.example" ".streamlit\secrets.toml"
    echo 🔑 Please edit .streamlit\secrets.toml with your OpenRouter API key
    pause
)

REM Run tests
echo 🔬 Running pre-flight tests...
python test_setup.py

if %errorlevel% equ 0 (
    echo 🚀 Starting Streamlit application...
    streamlit run app.py
) else (
    echo ❌ Tests failed. Please fix the issues before running.
    pause
)

pause
