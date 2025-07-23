@echo off
echo 🏥 Starting KKH Nursing Chatbot...

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Check if PDF exists
if not exist "data\KKH Information file.pdf" (
    echo ⚠️  Warning: PDF file not found at 'data\KKH Information file.pdf'
    echo Please ensure the PDF file is in the correct location.
)

REM Check if logo exists
if not exist "logo\photo_2025-06-16_15-57-21.jpg" (
    echo ⚠️  Warning: Logo file not found
)

REM Start the application
echo 🚀 Starting Streamlit application...
streamlit run app.py --server.port=8501 --server.address=0.0.0.0

echo ✅ Application started! Access at http://localhost:8501
pause
