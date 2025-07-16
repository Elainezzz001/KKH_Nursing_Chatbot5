# 🏥 KKH Nursing Chatbot

A comprehensive Streamlit application designed for KK Women's and Children's Hospital nursing staff, providing AI-powered assistance for pediatric nursing procedures, fluid calculations, and knowledge assessment.

## 🚀 Features

### 💬 Intelligent Chat Assistant
- **AI-Powered Responses**: Uses LM Studio with openhermes-2.5-mistral-7b model
- **Context-Aware**: Retrieves relevant information from KKH nursing documentation
- **Quick Prompts**: Pre-defined buttons for common queries:
  - Signs of dehydration
  - Paediatric CPR steps
  - Fluid resuscitation protocols
  - When to refer to doctor

### 💧 Pediatric Fluid Calculator
- **Maintenance Fluids**: Holliday-Segar method calculations
- **Dehydration Management**: 5% and 10% dehydration replacement
- **Shock Resuscitation**: Emergency fluid bolus calculations
- **Age/Weight Based**: Accurate dosing for pediatric patients

### 📋 Knowledge Assessment Quiz
- **15 Evidence-Based Questions**: Covering pediatric nursing essentials
- **Immediate Feedback**: Explanations for each answer
- **Score Tracking**: Performance monitoring and review
- **Topics Include**: CPR, fluid management, emergency protocols, vital signs

### 🎨 Professional UI
- **Modern Design**: Custom CSS with medical color scheme
- **Responsive Layout**: Sidebar navigation and organized sections
- **Chat Bubbles**: Clear conversation display
- **Visual Feedback**: Color-coded quiz results and calculations

## 🔧 Technical Stack

- **Frontend**: Streamlit with custom CSS
- **AI Model**: LM Studio (openhermes-2.5-mistral-7b)
- **Embeddings**: intfloat/multilingual-e5-large-instruct
- **Vector Search**: FAISS for document retrieval
- **PDF Processing**: PyPDF2 for content extraction
- **Deployment**: Docker + Fly.io

## 📋 Prerequisites

1. **LM Studio Setup**:
   - Install LM Studio from [lmstudio.ai](https://lmstudio.ai)
   - Download the `openhermes-2.5-mistral-7b` model
   - Start LM Studio server on `http://localhost:1234`

2. **Python Environment**:
   - Python 3.10 or higher
   - pip package manager

## 🚀 Local Development

### 1. Clone and Setup
```bash
git clone <repository-url>
cd "FYP Nursing Chatbot 5"
pip install -r requirements.txt
```

### 2. Start LM Studio
- Open LM Studio
- Load the `openhermes-2.5-mistral-7b` model
- Start the local server (default: http://localhost:1234)

### 3. Run the Application
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 🐳 Docker Deployment

### Build and Run Locally
```bash
docker build -t kkh-nursing-chatbot .
docker run -p 8080:8080 kkh-nursing-chatbot
```

**Note**: For Docker deployment, you'll need to set up LM Studio or an alternative model serving solution within the container or as a separate service.

## ☁️ Fly.io Deployment

### Prerequisites
```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Login to Fly.io
fly auth login
```

### Deploy
```bash
# Initialize (if not already done)
fly launch

# Deploy
fly deploy

# Check status
fly status
```

### Environment Configuration
The app expects LM Studio to be available at `http://localhost:1234`. For production deployment, you may need to:

1. **Option A**: Include LM Studio in the container
2. **Option B**: Use a cloud-based model API
3. **Option C**: Deploy LM Studio as a separate service

## 📁 File Structure

```
FYP Nursing Chatbot 5/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── fly.toml              # Fly.io deployment config
├── README.md             # This file
├── data/
│   └── KKH Information file.pdf  # Nursing documentation
└── logo/
    └── photo_2025-06-16_15-57-21.jpg  # App logo
```

## 🔒 Security & Privacy

- **Local Processing**: All AI processing happens locally
- **No External APIs**: No patient data sent to third parties
- **Offline Capable**: Works without internet connectivity
- **Secure**: Patient information remains within your infrastructure

## 📚 Usage Guide

### Chat Assistant
1. Select "💬 Chat" from the sidebar
2. Use quick prompt buttons for common questions
3. Type custom questions in the text input
4. The AI will search KKH documentation and provide context-aware responses

### Fluid Calculator
1. Select "💧 Fluid Calculator" from the sidebar
2. Enter patient weight and age
3. Choose clinical situation (maintenance, dehydration, shock)
4. Click "Calculate" for dosing recommendations

### Knowledge Quiz
1. Select "📋 Quiz" from the sidebar
2. Click "Start New Quiz" to begin
3. Answer multiple-choice questions
4. Review explanations and final score
5. Use "Review Answers" to see detailed feedback

## 🛠️ Customization

### Adding Quiz Questions
Edit the `QUIZ_QUESTIONS` list in `app.py`:
```python
QUIZ_QUESTIONS.append({
    "question": "Your question here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct": 0,  # Index of correct answer
    "explanation": "Explanation of the correct answer"
})
```

### Updating Documentation
Replace `data/KKH Information file.pdf` with updated nursing protocols. The app will automatically re-process the content.

### Modifying Calculations
Update the fluid calculation functions in `app.py`:
- `calculate_maintenance_fluid()`
- `calculate_dehydration_fluid()`
- `calculate_shock_fluid()`

## 🐛 Troubleshooting

### Common Issues

1. **LM Studio Connection Error**:
   - Ensure LM Studio is running on port 1234
   - Check that the model is loaded and server is started
   - Verify no firewall blocking the connection

2. **PDF Loading Issues**:
   - Ensure `KKH Information file.pdf` exists in the `data/` folder
   - Check PDF is readable and not password-protected
   - Verify sufficient memory for embedding processing

3. **Performance Issues**:
   - Increase Docker memory allocation
   - Use smaller embedding models if needed
   - Consider chunking large PDFs differently

### Logs and Debugging
```bash
# View Fly.io logs
fly logs

# Local debugging
streamlit run app.py --logger.level=debug
```

## 📞 Support

For technical issues:
1. Check the troubleshooting section above
2. Review application logs
3. Contact your IT department or system administrator

## ⚠️ Important Disclaimer

This tool is designed to assist healthcare professionals and should not replace:
- Clinical judgment
- Hospital protocols
- Direct physician consultation
- Emergency procedures

Always follow your institution's guidelines and seek appropriate medical supervision when needed.

## 📄 License

This application is developed for internal use at KKH. Please ensure compliance with your institution's software usage policies.

---

*KKH Nursing Chatbot v1.0 | Built for KKH Staff Use Only*
