#!/bin/bash

# KKH Nursing Chatbot Startup Script

echo "🏥 Starting KKH Nursing Chatbot..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check if PDF exists
if [ ! -f "data/KKH Information file.pdf" ]; then
    echo "⚠️  Warning: PDF file not found at 'data/KKH Information file.pdf'"
    echo "Please ensure the PDF file is in the correct location."
fi

# Check if logo exists
if [ ! -f "logo/photo_2025-06-16_15-57-21.jpg" ]; then
    echo "⚠️  Warning: Logo file not found"
fi

# Start the application
echo "🚀 Starting Streamlit application..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0

echo "✅ Application started! Access at http://localhost:8501"
