# 🏥 KKH Nursing Chatbot

An AI-powered chatbot designed specifically for nurses at KK Women's and Children's Hospital (KKH) to provide quick access to evidence-based information and clinical decision support.

## ✨ Features

- **🤖 AI Chat Assistant**: Get instant answers to clinical questions based on KKH guidelines
- **🧠 Knowledge Quiz**: Test and reinforce your understanding with interactive quizzes  
- **💧 Fluid Calculator**: Calculate pediatric fluid requirements using standard formulas
- **📖 Evidence-Based**: All responses are based on official KKH medical guidelines

## 🛠️ Technology Stack

- **Frontend**: Streamlit for interactive web interface
- **AI Model**: OpenHermes-2.5-Mistral-7B via LM Studio (local deployment)
- **Embeddings**: intfloat/multilingual-e5-large-instruct for semantic search
- **Document Processing**: PyPDF2 for extracting information from PDF guidelines

## 📋 Prerequisites

### For Local Development (Full Features)
1. **Python 3.8+** installed
2. **LM Studio** installed and running
   - Download from: https://lmstudio.ai/
   - Install OpenHermes-2.5-Mistral-7B model
   - Start server on `http://192.168.75.1:1234`

### For Deployment (Limited Features)
- Only Python 3.8+ required
- LLM features will show fallback messages

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "FYP Nursing Chatbot 5"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify file structure**
   ```
   /
   ├── app.py
   ├── requirements.txt
   ├── quiz_data.json
   ├── data/
   │   └── KKH Information file.pdf
   └── logo/
       └── photo_2025-06-16_15-57-21.jpg
   ```

## 🏃‍♂️ Running the Application

### Local Development
1. **Start LM Studio** (if you want full AI features)
   - Open LM Studio
   - Load OpenHermes-2.5-Mistral-7B model
   - Start server on `http://192.168.75.1:1234`

2. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

3. **Access the application**
   - Open your browser to `http://localhost:8501`

### Deployment on Render.com

1. **Connect your GitHub repository** to Render
2. **Create a new Web Service** with these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
   - **Environment Variable**: `RENDER=true`

## 📖 Usage Guide

### 💬 Chat Assistant
- Use predefined quick questions or type your own
- Get AI-powered responses based on KKH guidelines
- View source information for transparency
- Access recent question history

### 🧠 Knowledge Quiz
- Take a 15-question quiz based on KKH protocols
- Get immediate feedback with explanations
- Track your score and progress
- Restart anytime to practice

### 💧 Fluid Calculator
- Input patient weight, age, and clinical condition
- Get calculated daily fluid requirements
- View hourly rates and detailed breakdowns
- Includes condition-specific adjustments

## 🔧 Configuration

### LM Studio Setup
1. Download and install LM Studio
2. Install the OpenHermes-2.5-Mistral-7B model
3. Configure server settings:
   - Host: `192.168.75.1`
   - Port: `1234`
   - API endpoint: `/v1/chat/completions`

### PDF Processing
- The app automatically processes the PDF on first load
- Embeddings are cached for better performance
- Large PDFs may take a few minutes to process initially

## 🔒 Privacy & Security

- **Local Processing**: All data processing happens locally when possible
- **No Patient Data**: No patient information is stored or transmitted
- **Secure Guidelines**: Medical guidelines are processed locally for privacy
- **Offline Capable**: Core features work without internet connection

## 🐛 Troubleshooting

### Common Issues

1. **"Cannot connect to LM Studio"**
   - Ensure LM Studio is running on the correct port
   - Check that the model is loaded
   - Verify network connectivity

2. **PDF not processing**
   - Check that `KKH Information file.pdf` exists in the `data/` folder
   - Ensure the file is not corrupted
   - Try restarting the application

3. **Embedding model fails to load**
   - Check internet connection for first-time download
   - Ensure sufficient disk space (model is ~2GB)
   - Clear cache and retry

4. **Quiz not loading**
   - Check that `quiz_data.json` exists
   - Verify JSON formatting is correct
   - Application will fallback to default questions

### Performance Tips

- **First Load**: Initial PDF processing may take 2-3 minutes
- **Memory Usage**: Embedding model requires ~4GB RAM
- **Speed**: Local LM Studio responses depend on your hardware

## 🔄 Updates & Maintenance

### Adding New Questions
Edit `quiz_data.json` to add or modify quiz questions:
```json
{
  "question": "Your question here?",
  "options": {
    "A": "Option A",
    "B": "Option B", 
    "C": "Option C",
    "D": "Option D"
  },
  "correct_answer": "A",
  "explanation": "Explanation for the correct answer"
}
```

### Updating PDF Guidelines
1. Replace `data/KKH Information file.pdf` with new version
2. Delete `embeddings.pkl` and `chunks.pkl` if they exist
3. Restart the application to reprocess

## ⚠️ Important Disclaimers

- **Educational Purpose**: This tool is for educational and reference purposes only
- **Clinical Decisions**: Always follow official hospital protocols
- **Senior Consultation**: Consult with senior staff or physicians for critical clinical decisions
- **Not Diagnostic**: This tool does not provide medical diagnosis
- **Professional Judgment**: Use professional nursing judgment in all situations

## 📞 Support

For technical support or feedback:
- Contact the IT department
- Submit issues through the hospital's internal ticketing system
- For urgent technical issues during patient care, use backup protocols

## 📄 License

This application is developed for internal use at KK Women's and Children's Hospital. All rights reserved.

## 🙏 Acknowledgments

- KKH Nursing Team for protocol guidance
- IT Department for technical support
- OpenHermes team for the language model
- Sentence Transformers team for embedding models
