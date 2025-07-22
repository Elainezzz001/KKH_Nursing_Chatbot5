# KKH Nursing Chatbot 🏥

A comprehensive AI-powered chatbot designed specifically for nursing staff at KK Women's and Children's Hospital (KKH). This application provides instant access to nursing guidelines, interactive quizzes, fluid calculations, and AI-powered Q&A capabilities.

## Features

### 🤖 AI-Powered Chat
- **LLM Integration**: Uses Zephyr-7B-Beta model via OpenRouter for intelligent responses
- **RAG (Retrieval Augmented Generation)**: Semantic search through nursing documentation
- **Context-Aware**: Provides responses based on KKH nursing guidelines and protocols

### 📚 Knowledge Base
- **PDF Integration**: Automatically processes nursing guidelines from PDF documents
- **Semantic Search**: Uses `intfloat/multilingual-e5-large-instruct` for embedding generation
- **FAISS Vector Search**: Fast similarity search for relevant content retrieval

### 🧮 Fluid Calculator
- **Maintenance Fluid Calculation**: Holliday-Segar method implementation
- **Dehydration Replacement**: Calculates fluid deficits based on percentage dehydration
- **Age-Appropriate**: Considers patient weight and age for accurate calculations

### 📝 Interactive Quiz Module
- **15+ Nursing Questions**: Covers essential pediatric and maternal nursing topics
- **Immediate Feedback**: Shows correct answers and explanations
- **Score Tracking**: Monitors performance and provides detailed results
- **Restart Capability**: Allows multiple quiz attempts

### 🎯 Pre-defined Quick Questions
- Hypoglycemia treatment protocols
- CPR ratios for infants
- Normal vital signs for newborns
- Newborn resuscitation steps
- Breastfeeding guidelines

## Technical Stack

- **Frontend**: Streamlit (Mobile-friendly responsive design)
- **AI Model**: Zephyr-7B-Beta via OpenRouter API
- **Embeddings**: Sentence Transformers (multilingual-e5-large-instruct)
- **Vector Database**: FAISS for similarity search
- **PDF Processing**: PyMuPDF for text extraction
- **Deployment**: Docker + Fly.io ready

## Installation & Setup

### Prerequisites
- Python 3.11+
- OpenRouter API key (get from [OpenRouter](https://openrouter.ai/))

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd kkh-nursing-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.template .env
   # Edit .env and add your OPENROUTER_API_KEY
   ```

4. **Place your PDF file**
   - Add your nursing guidelines PDF to `data/KKH Information file.pdf`
   - The app will automatically process and embed the content

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`

### Environment Variables

```bash
OPENROUTER_API_KEY=your_api_key_here  # Required: Your OpenRouter API key
MODEL_NAME=HuggingFaceH4/zephyr-7b-beta  # Optional: Model specification
LM_STUDIO_URL=https://openrouter.ai/api/v1/chat/completions  # Optional: API endpoint
```

## Deployment

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t kkh-nursing-chatbot .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8501 -e OPENROUTER_API_KEY=your_key_here kkh-nursing-chatbot
   ```

### Fly.io Deployment

1. **Install Fly CLI**
   ```bash
   # Install flyctl (follow official Fly.io documentation)
   ```

2. **Deploy to Fly.io**
   ```bash
   fly deploy
   ```

3. **Set environment variables**
   ```bash
   fly secrets set OPENROUTER_API_KEY=your_api_key_here
   ```

## File Structure

```
kkh-nursing-chatbot/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── fly.toml                        # Fly.io deployment config
├── .env.template                   # Environment variables template
├── README.md                       # This file
├── data/
│   └── KKH Information file.pdf    # Nursing guidelines PDF
├── logo/
│   └── photo_2025-06-16_15-57-21.jpg  # Hospital logo
└── embeddings.pkl                  # Generated embeddings (auto-created)
```

## Usage Guide

### Chat Interface
1. **Ask Questions**: Type nursing-related questions in the chat input
2. **Quick Questions**: Use sidebar buttons for common queries
3. **Context-Aware Responses**: The AI uses your PDF content to provide accurate answers

### Quiz Module
1. **Start Quiz**: Click "Start Quiz" in the sidebar
2. **Answer Questions**: Select from multiple choice options
3. **View Results**: Get detailed feedback and explanations
4. **Restart**: Take the quiz multiple times to improve knowledge

### Fluid Calculator
1. **Maintenance Fluids**: Enter patient weight for daily fluid requirements
2. **Dehydration**: Calculate replacement fluids based on dehydration percentage
3. **Clinical Guidelines**: Follow Holliday-Segar method recommendations

## API Integration

The application integrates with OpenRouter to access the Zephyr-7B-Beta model:

```python
# Example API call structure
{
    "model": "HuggingFaceH4/zephyr-7b-beta",
    "messages": [
        {"role": "system", "content": "nursing_assistant_prompt"},
        {"role": "user", "content": "user_question_with_context"}
    ],
    "temperature": 0.7,
    "max_tokens": 500
}
```

## Customization

### Adding New Quiz Questions
Edit the `QuizModule` class in `app.py`:

```python
{
    "question": "Your question here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct": 0,  # Index of correct answer (0-3)
    "explanation": "Explanation of the correct answer"
}
```

### Modifying Fluid Calculations
Adjust the `FluidCalculator` class methods:
- `calculate_maintenance_fluid()`: For maintenance fluid protocols
- `calculate_dehydration_fluid()`: For replacement fluid calculations

### Changing AI Model
Update the model configuration in the constants section:

```python
MODEL_NAME = "your_preferred_model"  # Change to different OpenRouter model
LM_STUDIO_URL = "your_api_endpoint"  # Change if using different service
```

## Troubleshooting

### Common Issues

1. **PDF Processing Errors**
   - Ensure PDF is not password-protected
   - Check file path: `data/KKH Information file.pdf`
   - Verify PDF contains extractable text

2. **API Key Issues**
   - Verify OPENROUTER_API_KEY is set correctly
   - Check API key permissions on OpenRouter dashboard
   - Ensure sufficient API credits

3. **Embedding Model Loading**
   - First run may take time to download the embedding model
   - Ensure sufficient disk space (~2GB for model)
   - Check internet connection for model download

4. **Memory Issues**
   - Increase Docker memory allocation if using containers
   - Consider using smaller embedding models for resource-constrained environments

### Performance Optimization

- **Embedding Caching**: Embeddings are cached in `embeddings.pkl`
- **Model Loading**: Embedding model is loaded once and reused
- **Chunk Size**: Adjust `CHUNK_SIZE` constant for different performance/accuracy trade-offs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is developed for KK Women's and Children's Hospital nursing staff. Please ensure compliance with hospital policies and data protection regulations.

## Support

For technical issues or feature requests:
1. Check the troubleshooting section above
2. Review application logs for error details
3. Ensure all dependencies are correctly installed
4. Verify environment variables are properly set

## Security Considerations

- **API Keys**: Never commit API keys to version control
- **PDF Content**: Ensure sensitive patient information is not included in training documents
- **Access Control**: Consider implementing authentication for production deployments
- **Data Privacy**: Follow hospital data protection policies

---

**Note**: This application is designed for educational and clinical support purposes. Always follow hospital protocols and consult with senior staff for critical patient care decisions.
