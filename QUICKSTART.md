# KKH Nursing Chatbot - Quick Start Guide

## 🚀 Quick Start (Windows)

### Option 1: Automated Setup
```cmd
python setup.py
run_local.bat
```

### Option 2: Manual Setup
```cmd
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key
copy secrets.toml.example .streamlit\secrets.toml
# Edit .streamlit\secrets.toml with your OpenRouter API key

# 4. Run the application
streamlit run app.py
```

## 🔑 API Key Setup

1. Go to [OpenRouter](https://openrouter.ai/)
2. Sign up and get your API key
3. Edit `.streamlit\secrets.toml`:
   ```toml
   OPENROUTER_API_KEY = "your-actual-api-key-here"
   ```

## 🧪 Testing

Run pre-deployment tests:
```cmd
python test_setup.py
```

## ☁️ Deployment to Fly.io

1. Install [Fly.io CLI](https://fly.io/docs/hands-on/install-flyctl/)
2. Login: `flyctl auth login`
3. Deploy: `deploy.bat` (Windows) or `./deploy.sh` (Linux/Mac)

## 📱 Application Features

- **Chat Assistant**: AI-powered responses based on nursing PDF
- **Fluid Calculator**: Holliday-Segar method for pediatric patients
- **Quiz Module**: 15 multiple-choice nursing questions
- **Quick Prompts**: Common nursing question buttons

## 🏗️ Project Structure

```
kkh-nursing-chatbot/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── setup.py                 # Development environment setup
├── test_setup.py            # Pre-deployment tests
├── run_local.bat            # Local development (Windows)
├── deploy.bat/deploy.sh     # Deployment scripts
├── Dockerfile               # Container configuration
├── fly.toml                 # Fly.io deployment config
├── .streamlit/
│   ├── config.toml          # Streamlit configuration
│   └── secrets.toml         # API keys (local only)
├── data/
│   └── KKH Information file.pdf  # Nursing guide PDF
└── logo/
    └── photo_2025-06-16_15-57-21.jpg  # App logo
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Run `pip install -r requirements.txt`
2. **PDF Not Found**: Ensure `data/KKH Information file.pdf` exists
3. **API Errors**: Check your OpenRouter API key
4. **Port Already in Use**: Kill other Streamlit processes

### Environment Variables

For production deployment, set:
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `PORT`: Application port (default: 8080)

## 📞 Support

- Check `README.md` for detailed documentation
- Run `python test_setup.py` for diagnostics
- Review logs with `flyctl logs` for Fly.io deployments
