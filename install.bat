@echo off
setlocal

echo ================================================================
echo KKH Nursing Chatbot - Installation Script
echo ================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

echo Installing required packages...
echo This may take several minutes on first run...
echo.

REM Install requirements
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install some packages
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo ================================================================
echo Installation completed successfully!
echo ================================================================
echo.

echo Next steps:
echo 1. Start LM Studio from https://lmstudio.ai/
echo 2. Load the OpenHermes 2.5 Mistral 7B model
echo 3. Start the local server on port 1234
echo 4. Ensure your PDF file is in the data/ folder
echo 5. Run the test script: python test_setup.py
echo 6. Start the chatbot: streamlit run app.py
echo.

echo Would you like to run the test script now? (y/n)
set /p choice="Enter your choice: "

if /i "%choice%"=="y" (
    echo.
    echo Running system tests...
    python test_setup.py
    echo.
    echo Test completed. Check results above.
    echo.
    echo If all tests passed, you can start the chatbot with:
    echo    streamlit run app.py
)

echo.
echo Setup complete! Press any key to exit...
pause >nul
