@echo off
echo 🏥 KKH Nursing Chatbot - Windows Startup Script
echo ================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.11+ and try again
    pause
    exit /b 1
)

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not installed
    echo Please install pip and try again
    pause
    exit /b 1
)

echo ✅ Python and pip found

REM Install dependencies if not already installed
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Check if .env file exists
if not exist .env (
    echo ⚠️  .env file not found
    echo Creating .env file from template...
    copy .env.template .env
    echo.
    echo 🔑 Please edit .env file and add your OPENROUTER_API_KEY
    echo Then run this script again
    pause
    exit /b 1
)

REM Run setup test
echo 🧪 Running setup test...
python test_setup.py
if errorlevel 1 (
    echo ❌ Setup test failed
    pause
    exit /b 1
)

echo.
echo 🚀 Starting KKH Nursing Chatbot...
echo 🌐 The application will open in your default browser
echo 🛑 Press Ctrl+C to stop the application
echo.

REM Start Streamlit
streamlit run app.py

pause
