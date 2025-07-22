#!/bin/bash

echo "🏥 KKH Nursing Chatbot - Startup Script"
echo "========================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.11+ and try again"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed"
    echo "Please install pip3 and try again"
    exit 1
fi

echo "✅ Python and pip found"

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found"
    echo "Creating .env file from template..."
    cp .env.template .env
    echo ""
    echo "🔑 Please edit .env file and add your OPENROUTER_API_KEY"
    echo "Then run this script again"
    exit 1
fi

# Run setup test
echo "🧪 Running setup test..."
python3 test_setup.py
if [ $? -ne 0 ]; then
    echo "❌ Setup test failed"
    exit 1
fi

echo ""
echo "🚀 Starting KKH Nursing Chatbot..."
echo "🌐 The application will open in your default browser"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

# Start Streamlit
streamlit run app.py
