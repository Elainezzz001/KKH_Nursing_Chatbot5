#!/bin/bash

# KKH Nursing Chatbot - Development Setup Script

echo "🏥 KKH Nursing Chatbot - Development Setup"
echo "=========================================="

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.10 or higher."
    exit 1
fi

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "✅ Python version: $python_version"

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

echo "✅ pip is available"

# Install requirements
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Check if data file exists
if [ -f "data/KKH Information file.pdf" ]; then
    echo "✅ KKH Information file found"
else
    echo "⚠️  KKH Information file.pdf not found in data/ folder"
    echo "   Please ensure the PDF is placed in the data/ directory"
fi

echo ""
echo "🚀 Setup complete! To run the application:"
echo ""
echo "1. Start LM Studio:"
echo "   - Open LM Studio"
echo "   - Load the openhermes-2.5-mistral-7b model"
echo "   - Start the server on http://localhost:1234"
echo ""
echo "2. Run the Streamlit app:"
echo "   streamlit run app.py"
echo ""
echo "3. Open your browser to http://localhost:8501"
echo ""
echo "📚 For more information, see README.md"
