@echo off
echo Starting KKH Nursing Chatbot...
echo.
echo Please ensure LM Studio is running at http://192.168.75.1:1234
echo.
pause
echo.
echo Installing dependencies (if needed)...
pip install -r requirements.txt
echo.
echo Starting Streamlit application...
streamlit run app.py
pause
