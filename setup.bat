@echo off
echo 🏥 KKH Nursing Chatbot - Development Setup
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.10 or higher.
    pause
    exit /b 1
)

echo ✅ Python is available

REM Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip is not installed. Please install pip.
    pause
    exit /b 1
)

echo ✅ pip is available

REM Install requirements
echo 📦 Installing Python dependencies...
pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo ✅ Dependencies installed successfully
) else (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Check if data file exists
if exist "data\KKH Information file.pdf" (
    echo ✅ KKH Information file found
) else (
    echo ⚠️  KKH Information file.pdf not found in data\ folder
    echo    Please ensure the PDF is placed in the data\ directory
)

echo.
echo 🚀 Setup complete! To run the application:
echo.
echo 1. Start LM Studio:
echo    - Open LM Studio
echo    - Load the openhermes-2.5-mistral-7b model
echo    - Start the server on http://localhost:1234
echo.
echo 2. Run the Streamlit app:
echo    streamlit run app.py
echo.
echo 3. Open your browser to http://localhost:8501
echo.
echo 📚 For more information, see README.md
echo.
pause
