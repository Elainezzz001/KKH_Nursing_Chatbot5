# KKH Nursing Chatbot

A comprehensive Streamlit-based chatbot application designed for KKH nursing professionals, featuring AI-powered semantic search, fluid calculators, and knowledge quizzes.

## Features

### 🔍 Semantic Search Chatbot
- Natural language question processing
- AI-powered responses using OpenHermes 2.5 Mistral 7B via LM Studio
- Context-aware answers based on embedded PDF knowledge
- Quick prompt buttons for common queries

### 📋 Fluid Calculator
- Holliday-Segar method for maintenance fluid calculations
- Dehydration deficit calculations (5% and 10%)
- Shock resuscitation protocols
- Clinical guidance and monitoring tips

### ❓ Knowledge Quiz
- 15 comprehensive pediatric nursing questions
- Immediate feedback with explanations
- Score tracking and answer review
- Restart functionality for practice

## Technical Stack

- **Frontend**: Streamlit
- **LLM**: OpenHermes 2.5 Mistral 7B (via LM Studio)
- **Embeddings**: Multilingual E5 Large Instruct (SentenceTransformer)
- **Vector Search**: FAISS indexing
- **PDF Processing**: PyPDF2

## Setup Instructions

### Prerequisites

1. **Python 3.8 or higher**
2. **LM Studio** running locally at `http://192.168.75.1:1234`
   - Install LM Studio from [lmstudio.ai](https://lmstudio.ai)
   - Load the OpenHermes 2.5 Mistral 7B model
   - Start the local server on port 1234

### Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify PDF file**:
   - Ensure `KKH Information file.pdf` is in the `data/` folder
   - The app will automatically process and index this PDF on startup

### Running the Application

1. **Start LM Studio** and ensure the server is running at `http://192.168.75.1:1234`

2. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** and navigate to `http://localhost:8501`

## Usage Guide

### Semantic Search Chat
1. Select "🔍 Semantic Search Chat" from the sidebar
2. Use quick prompt buttons or type your own questions
3. The app will search the PDF content and provide AI-generated responses

### Fluid Calculator
1. Select "📋 Fluid Calculator" from the sidebar
2. Enter patient weight and age
3. Choose clinical situation (Maintenance, Dehydration, Shock)
4. Click "Calculate Fluid Requirements" for results

### Knowledge Quiz
1. Select "❓ Knowledge Quiz" from the sidebar
2. Click "🚀 Start Quiz" to begin
3. Answer questions and receive immediate feedback
4. Review your answers at the end
5. Use "🔄 Restart Quiz" to practice again

## Configuration

### LM Studio Settings
- **URL**: `http://192.168.75.1:1234/v1/chat/completions`
- **Model**: `openhermes-2.5-mistral-7b`
- **Temperature**: 0.7
- **Max Tokens**: 500

### Embedding Model
- **Model**: `intfloat/multilingual-e5-large-instruct`
- **Chunk Size**: 500 characters
- **Search Results**: Top 3 relevant chunks

## Troubleshooting

### Common Issues

1. **"Cannot connect to LM Studio"**
   - Ensure LM Studio is running
   - Check the IP address and port (192.168.75.1:1234)
   - Verify the model is loaded in LM Studio

2. **"PDF file not found"**
   - Ensure `KKH Information file.pdf` is in the `data/` folder
   - Check file name matches exactly

3. **Slow embedding model loading**
   - First-time loading may take several minutes
   - Subsequent runs will be faster due to caching

4. **Memory issues**
   - Ensure sufficient RAM (minimum 8GB recommended)
   - Close other applications if needed

### Error Messages

- **Connection Error**: Check LM Studio is running and accessible
- **Embedding Error**: Verify internet connection for model download
- **PDF Error**: Check PDF file exists and is readable

## Customization

### Adding New Quiz Questions
Edit the `generate_quiz_questions()` method in the `KKHChatbot` class to add more questions.

### Modifying Quick Prompts
Update the `quick_prompts` list in the semantic search section.

### Changing Styling
Modify the CSS in the `st.markdown()` section at the top of the file.

## System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space for models
- **Network**: Internet connection for initial model download
- **OS**: Windows, macOS, or Linux

## Support

For technical issues or content updates, please contact the development team.

## License

This application is developed for KKH internal use. All medical content should be verified with current clinical guidelines.
