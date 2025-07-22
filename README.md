# KKH Nursing Chatbot

A comprehensive Streamlit-based nursing chatbot web application designed for pediatric and neonatal care guidance.

## Features

🤖 **AI-Powered Chat Assistant**
- Uses semantic search with FAISS vector store
- Searches nursing PDF content for relevant information
- Powered by OpenRouter API with advanced language models
- Provides accurate, context-aware responses

🧮 **Fluid Calculator**
- Holliday-Segar method for maintenance fluid calculation
- Supports pediatric patients from 0.1kg to 150kg
- Provides daily total, hourly rate, and safe ranges

📝 **Interactive Quiz Module**
- 15 multiple-choice questions on nursing knowledge
- Real-time scoring and feedback
- Questions cover pediatric/neonatal care topics
- Restart functionality for repeated practice

💡 **Quick Prompt Suggestions**
- Pre-defined common nursing questions
- One-click to ask frequent queries
- Covers heart rates, medications, developmental milestones

## Technology Stack

- **Frontend**: Streamlit
- **AI Model**: OpenRouter API (WizardLM-2-8x22b)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **PDF Processing**: PyMuPDF
- **Deployment**: Fly.io

## Local Development

### Prerequisites
- Python 3.11+
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd kkh-nursing-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**
   - Sign up at [OpenRouter](https://openrouter.ai/)
   - Get your API key
   - Update `.streamlit/secrets.toml`:
     ```toml
     OPENROUTER_API_KEY = "your-actual-api-key"
     ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the app**
   - Open http://localhost:8501 in your browser

## Deployment to Fly.io

### Prerequisites
- [Fly.io CLI](https://fly.io/docs/hands-on/install-flyctl/) installed
- Fly.io account

### Deploy Steps

1. **Login to Fly.io**
   ```bash
   flyctl auth login
   ```

2. **Set secrets**
   ```bash
   flyctl secrets set OPENROUTER_API_KEY="your-actual-api-key"
   ```

3. **Deploy the application**
   ```bash
   flyctl deploy
   ```

4. **Open your deployed app**
   ```bash
   flyctl open
   ```

## Project Structure

```
kkh-nursing-chatbot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── fly.toml              # Fly.io deployment config
├── data/
│   └── KKH Information file.pdf  # Nursing guide PDF
├── logo/
│   └── photo_2025-06-16_15-57-21.jpg  # App logo
├── .streamlit/
│   ├── config.toml       # Streamlit configuration
│   └── secrets.toml      # API keys (local only)
└── README.md            # This file
```

## Usage Guide

### Chat Interface
1. Type your nursing question in the chat input
2. Use quick prompt buttons for common questions
3. View responses based on the nursing PDF content
4. Chat history is maintained during the session

### Fluid Calculator
1. Click "Toggle Calculator" in the sidebar
2. Enter patient weight in kg
3. Click "Calculate" to get maintenance fluid requirements
4. Results show daily total, hourly rate, and safe ranges

### Quiz Module
1. Click "Start Quiz" in the sidebar
2. Answer 15 multiple-choice questions
3. Get immediate scoring and feedback
4. Use "Reset Quiz" to start over

### Navigation
- **New Chat**: Clears chat history and returns to main interface
- **Start Quiz**: Begins the quiz module
- **Toggle Calculator**: Shows/hides the fluid calculator
- **Chat History**: Shows recent conversation topics

## Configuration

### API Configuration
The app uses OpenRouter API as a proxy to various language models. Configure your API key in:
- Local: `.streamlit/secrets.toml`
- Production: Fly.io secrets

### Model Settings
- **Chat Model**: microsoft/wizardlm-2-8x22b
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS with L2 distance
- **PDF Chunking**: ~500 character chunks with sentence boundaries

## Troubleshooting

### Common Issues

1. **PDF Not Loading**
   - Ensure `data/KKH Information file.pdf` exists
   - Check file permissions

2. **API Errors**
   - Verify OpenRouter API key is correct
   - Check internet connectivity
   - Ensure you have API credits

3. **Slow Performance**
   - First run will be slower due to PDF processing
   - Subsequent runs use cached embeddings
   - Consider using a lighter embedding model for faster startup

### Performance Optimization
- Embeddings are cached after first creation
- Vector store is persisted to disk
- Use `@st.cache_resource` for model loading

## Security Notes

- API keys are stored securely in Streamlit secrets
- No data is stored persistently except embeddings cache
- All conversations are session-based only

## License

This project is developed for educational purposes in nursing care.

## Support

For issues or questions, please check the troubleshooting section or contact the development team.
