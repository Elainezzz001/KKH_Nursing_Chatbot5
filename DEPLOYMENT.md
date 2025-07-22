# KKH Nursing Chatbot - Deployment Guide 🚀

This guide covers different deployment options for the KKH Nursing Chatbot.

## Quick Start (Local Development)

### Option 1: Using Startup Scripts

**Windows:**
```cmd
# Double-click start.bat or run in command prompt
start.bat
```

**Linux/macOS:**
```bash
# Make executable and run
chmod +x start.sh
./start.sh
```

### Option 2: Manual Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   # Copy template and edit
   cp .env.template .env
   # Edit .env file and add your OPENROUTER_API_KEY
   ```

3. **Test setup:**
   ```bash
   python test_setup.py
   ```

4. **Run application:**
   ```bash
   streamlit run app.py
   ```

## Production Deployment

### 1. Fly.io Deployment (Recommended)

Fly.io provides excellent global CDN and auto-scaling capabilities.

#### Prerequisites
- Fly.io account (free tier available)
- Flyctl CLI installed

#### Steps

1. **Install Flyctl:**
   ```bash
   # macOS
   brew install flyctl
   
   # Windows (PowerShell)
   powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
   
   # Linux
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login to Fly.io:**
   ```bash
   fly auth login
   ```

3. **Deploy application:**
   ```bash
   fly deploy
   ```

4. **Set environment variables:**
   ```bash
   fly secrets set OPENROUTER_API_KEY=your_actual_api_key_here
   ```

5. **Monitor deployment:**
   ```bash
   fly logs
   ```

6. **Open application:**
   ```bash
   fly open
   ```

#### Fly.io Configuration

The `fly.toml` file is already configured with:
- Singapore region (closest to KKH)
- Auto-scaling (0-1 machines)
- Health checks
- HTTPS termination

### 2. Docker Deployment

#### Local Docker

1. **Build image:**
   ```bash
   docker build -t kkh-nursing-chatbot .
   ```

2. **Run container:**
   ```bash
   docker run -p 8501:8501 \
     -e OPENROUTER_API_KEY=your_key_here \
     kkh-nursing-chatbot
   ```

#### Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  kkh-chatbot:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
    volumes:
      - ./data:/app/data
      - ./logo:/app/logo
```

Run with:
```bash
docker-compose up -d
```

### 3. Cloud Platforms

#### Google Cloud Run

1. **Build and push to Google Container Registry:**
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/kkh-chatbot
   ```

2. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy kkh-chatbot \
     --image gcr.io/YOUR_PROJECT_ID/kkh-chatbot \
     --platform managed \
     --region asia-southeast1 \
     --set-env-vars OPENROUTER_API_KEY=your_key_here
   ```

#### AWS ECS/Fargate

1. **Create ECR repository:**
   ```bash
   aws ecr create-repository --repository-name kkh-chatbot
   ```

2. **Build and push:**
   ```bash
   docker tag kkh-nursing-chatbot:latest \
     YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/kkh-chatbot:latest
   docker push YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/kkh-chatbot:latest
   ```

3. **Deploy using ECS/Fargate** (use AWS Console or CLI)

#### Azure Container Instances

```bash
az container create \
  --resource-group myResourceGroup \
  --name kkh-chatbot \
  --image your-registry/kkh-chatbot:latest \
  --dns-name-label kkh-chatbot \
  --ports 8501 \
  --environment-variables OPENROUTER_API_KEY=your_key_here
```

## Environment Variables

### Required
- `OPENROUTER_API_KEY`: Your OpenRouter API key

### Optional
- `MODEL_NAME`: AI model to use (default: HuggingFaceH4/zephyr-7b-beta)
- `LM_STUDIO_URL`: API endpoint (default: OpenRouter)

## Security Considerations

### API Key Management
- Never commit API keys to version control
- Use environment variables or secrets management
- Rotate API keys regularly

### Access Control
For production deployments, consider adding:

1. **Basic Authentication:**
   ```python
   # Add to app.py
   import streamlit_authenticator as stauth
   ```

2. **IP Whitelisting:**
   ```yaml
   # In fly.toml or cloud provider config
   allowed_ips:
     - "hospital_ip_range"
   ```

3. **SSL/TLS:**
   - Fly.io provides automatic HTTPS
   - For other platforms, ensure SSL certificates

### Data Protection
- PDF content should not contain patient information
- Consider encryption for sensitive configuration
- Follow hospital data protection policies

## Monitoring & Maintenance

### Health Checks
The application includes health check endpoints:
- Streamlit: `/_stcore/health`
- Custom health check available in the app

### Logging
Monitor application logs:
```bash
# Fly.io
fly logs

# Docker
docker logs container_name

# Cloud platforms
# Use respective logging services
```

### Updates
To update the application:

1. **Update code:**
   ```bash
   git pull origin main
   ```

2. **Redeploy:**
   ```bash
   # Fly.io
   fly deploy
   
   # Docker
   docker build -t kkh-nursing-chatbot . && docker-compose up -d
   ```

## Performance Optimization

### Memory Management
- Default: 1GB RAM (sufficient for most usage)
- Scale up if processing large PDFs or high concurrent users

### Embedding Cache
- Embeddings are cached in `embeddings.pkl`
- Persists across container restarts when using volumes

### Model Loading
- Embedding model downloads once (~2GB)
- Consider using init containers for faster startup

## Troubleshooting

### Common Issues

1. **API Key Errors:**
   ```bash
   # Check if API key is set
   fly secrets list
   # Set API key
   fly secrets set OPENROUTER_API_KEY=new_key
   ```

2. **Memory Issues:**
   ```toml
   # Increase memory in fly.toml
   [[vm]]
   memory_mb = 2048
   ```

3. **PDF Processing:**
   - Ensure PDF is not corrupted
   - Check file permissions
   - Verify PDF contains extractable text

4. **Model Download Issues:**
   - Check internet connectivity
   - Verify sufficient disk space
   - Consider using model mirrors

### Support Commands

```bash
# Check deployment status
fly status

# View recent logs
fly logs

# SSH into running instance
fly ssh console

# Restart application
fly restart

# Check resource usage
fly machine list
```

## Cost Optimization

### Fly.io
- Free tier: 3 shared CPUs, 256MB RAM
- Paid: ~$5-10/month for basic usage
- Auto-scaling reduces costs during low usage

### API Costs
- OpenRouter: Pay-per-token usage
- Typical cost: $0.01-0.10 per conversation
- Monitor usage in OpenRouter dashboard

### Optimization Tips
1. Cache responses when possible
2. Limit max_tokens in API calls
3. Use efficient chunking strategies
4. Monitor API usage patterns

---

## Quick Reference

### Essential Commands
```bash
# Test setup
python test_setup.py

# Run locally
streamlit run app.py

# Deploy to Fly.io
fly deploy

# Set secrets
fly secrets set OPENROUTER_API_KEY=your_key

# View logs
fly logs

# Open app
fly open
```

### Important Files
- `app.py`: Main application
- `requirements.txt`: Dependencies
- `Dockerfile`: Container configuration
- `fly.toml`: Deployment configuration
- `.env`: Environment variables (local)

For additional support, refer to the main README.md or check the troubleshooting section.
