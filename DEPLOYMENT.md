# 🚀 Fly.io Deployment Guide

This guide walks you through deploying the KKH Nursing Chatbot to Fly.io.

## Prerequisites

1. **Fly.io Account**: Sign up at [fly.io](https://fly.io)
2. **Fly CLI**: Install the Fly command-line tool
3. **Docker**: Ensure Docker is installed and running

## Installation Steps

### 1. Install Fly CLI

**Windows (PowerShell):**
```powershell
iwr https://fly.io/install.ps1 -useb | iex
```

**macOS/Linux:**
```bash
curl -L https://fly.io/install.sh | sh
```

### 2. Login to Fly.io
```bash
fly auth login
```

### 3. Initialize the App (if not already done)
```bash
cd "c:\FYP Nursing Chatbot 5"
fly launch
```

When prompted:
- Choose your app name (or use "kkh-nursing-chatbot")
- Select your preferred region (Singapore: sin)
- Choose not to set up PostgreSQL database
- Choose not to deploy immediately

### 4. Configure the App

The `fly.toml` file is already configured with:
- **Port**: 8080 (Streamlit default)
- **Memory**: 2GB (for ML models)
- **Region**: Singapore (sin)
- **Health checks**: Streamlit health endpoint

### 5. Deploy the Application
```bash
fly deploy
```

### 6. Check Deployment Status
```bash
fly status
fly logs
```

### 7. Open the Application
```bash
fly open
```

## Important Notes for Production

### 🤖 Model Serving Considerations

The current setup expects LM Studio to be running locally. For production deployment, you have several options:

#### Option A: Include Model in Container
```dockerfile
# Add to Dockerfile
RUN pip install transformers torch
COPY models/ /app/models/
```

#### Option B: Use Cloud Model APIs
Update `app.py` to use cloud-based APIs like:
- Azure OpenAI
- AWS Bedrock
- Google Vertex AI

#### Option C: Separate Model Service
Deploy LM Studio or similar as a separate Fly.io app and update the URL.

### 🔧 Environment Variables

Set sensitive configuration via Fly secrets:
```bash
fly secrets set MODEL_URL=http://your-model-service.fly.dev
fly secrets set API_KEY=your-secret-key
```

### 📊 Monitoring

Enable Fly.io monitoring:
```bash
fly dashboard
```

Monitor:
- CPU and memory usage
- Response times
- Error rates
- Health check status

### 🔄 Auto-scaling

Configure auto-scaling in `fly.toml`:
```toml
[machine]
  memory = "2gb"
  cpu_kind = "shared"
  cpus = 2

[http_service]
  min_machines_running = 1
  max_machines_running = 5
```

### 💾 Persistent Storage

If you need to store data:
```bash
fly volumes create data_volume --size 10gb
```

Update `fly.toml`:
```toml
[mounts]
  source = "data_volume"
  destination = "/app/data"
```

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Increase memory in `fly.toml`
   - Use smaller ML models
   - Optimize embedding processing

2. **Slow Startup**:
   - Pre-download models in Dockerfile
   - Use model caching
   - Optimize image size

3. **Connection Timeouts**:
   - Increase health check timeouts
   - Optimize app startup time
   - Use proper logging

### Debugging Commands

```bash
# View logs
fly logs --app kkh-nursing-chatbot

# SSH into running machine
fly ssh console

# Check machine status
fly machine list

# Scale machines
fly machine clone
```

### Performance Optimization

1. **Docker Image Size**:
   - Use multi-stage builds
   - Remove unnecessary dependencies
   - Use smaller base images

2. **ML Model Loading**:
   - Cache models in Docker layer
   - Use model quantization
   - Lazy load models

3. **Streamlit Performance**:
   - Use `@st.cache_resource` for expensive operations
   - Optimize PDF processing
   - Implement pagination for large datasets

## Security Considerations

1. **Secrets Management**:
   ```bash
   fly secrets set SECRET_KEY=your-secret-key
   fly secrets set DATABASE_URL=your-db-url
   ```

2. **Network Security**:
   - Use HTTPS (automatically enabled)
   - Configure IP restrictions if needed
   - Implement authentication if required

3. **Data Privacy**:
   - Ensure HIPAA compliance if handling patient data
   - Use encrypted storage
   - Implement audit logging

## Cost Optimization

1. **Machine Sizing**:
   - Start with smaller machines
   - Scale based on actual usage
   - Use auto-stop for development

2. **Regional Deployment**:
   - Deploy close to users
   - Consider data residency requirements

3. **Monitoring Costs**:
   ```bash
   fly dashboard
   ```
   Check usage in the billing section.

## Backup and Recovery

1. **Code Backup**:
   - Use Git for version control
   - Regular commits and tags

2. **Data Backup**:
   - Export volumes regularly
   - Document restoration procedures

3. **Configuration Backup**:
   - Save `fly.toml` and secrets
   - Document environment setup

---

**Need Help?**
- Fly.io Documentation: https://fly.io/docs/
- Community Forum: https://community.fly.io/
- Support: https://fly.io/support/
