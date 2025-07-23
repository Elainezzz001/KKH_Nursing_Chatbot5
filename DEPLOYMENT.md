# 🚀 Deployment Guide for Render.com

This guide will help you deploy the KKH Nursing Chatbot to Render.com for cloud hosting.

## 📋 Prerequisites

1. **GitHub Account** with your code repository
2. **Render.com Account** (free tier available)
3. **Repository Structure** as outlined in README.md

## 🔧 Deployment Steps

### Step 1: Prepare Your Repository

1. **Ensure all files are committed** to your GitHub repository:
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Verify file structure**:
   ```
   /
   ├── app.py
   ├── requirements.txt
   ├── quiz_data.json
   ├── Dockerfile (optional)
   ├── data/
   │   └── KKH Information file.pdf
   └── logo/
       └── photo_2025-06-16_15-57-21.jpg
   ```

### Step 2: Create Render Service

1. **Login to Render.com**
   - Go to https://render.com
   - Sign up or login with your GitHub account

2. **Create New Web Service**
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Select the repository containing your chatbot

### Step 3: Configure Build Settings

**Environment:**
- Runtime: `Python 3`

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
```

### Step 4: Set Environment Variables

Add these environment variables in Render dashboard:

| Key | Value | Description |
|-----|-------|-------------|
| `RENDER` | `true` | Tells app it's in deployed mode |
| `PYTHONUNBUFFERED` | `1` | Ensures logs are visible |

### Step 5: Advanced Settings

**Instance Type:**
- For development: `Free` (512 MB RAM)
- For production: `Starter` or higher (recommended for better performance)

**Auto-Deploy:**
- Enable auto-deploy for automatic updates when you push to GitHub

**Health Check Path:**
- Leave default or set to `/_stcore/health`

## 🎯 Expected Deployment Behavior

### ✅ Available Features on Render:
- **Knowledge Quiz**: Fully functional with all 15 questions
- **Fluid Calculator**: Complete pediatric fluid calculations
- **PDF Processing**: Automatic text extraction and chunking
- **Semantic Search**: Local embedding-based search
- **UI Interface**: Complete Streamlit interface

### ⚠️ Limited Features on Render:
- **AI Chat Responses**: Will show fallback message:
  *"This feature is available only on local LM Studio instance"*
- **Dynamic Question Generation**: Uses pre-defined quiz questions

## 🔍 Troubleshooting

### Common Deployment Issues

1. **Build Fails - Memory Error**
   ```
   Solution: Upgrade to Starter plan (1GB RAM minimum)
   The embedding model requires significant memory
   ```

2. **App Crashes - PDF Not Found**
   ```
   Error: PDF file not found at data/KKH Information file.pdf
   Solution: Ensure PDF is committed to Git and in correct folder
   ```

3. **Slow Initial Load**
   ```
   Cause: First-time embedding model download (~2GB)
   Solution: Be patient, subsequent loads will be faster
   ```

4. **Streamlit Port Issues**
   ```
   Error: App not accessible after deployment
   Solution: Ensure start command uses $PORT variable
   ```

### Performance Optimization

1. **Reduce Build Time**
   - Keep requirements.txt minimal
   - Use .gitignore to exclude unnecessary files

2. **Improve Loading Speed**
   - Consider pre-computing embeddings
   - Cache model files if possible

3. **Memory Management**
   - Monitor memory usage in Render dashboard
   - Upgrade plan if needed

## 📊 Monitoring Your Deployment

### Render Dashboard Features:
- **Logs**: Real-time application logs
- **Metrics**: CPU, memory, and request metrics
- **Health**: Service health status
- **Events**: Deployment history

### Key Metrics to Watch:
- **Memory Usage**: Should not exceed 80% consistently
- **Response Time**: Initial load may be slow, subsequent requests faster
- **Error Rate**: Monitor for PDF processing errors

## 🔄 Updates and Maintenance

### Updating Your App:
1. Make changes to your local code
2. Commit and push to GitHub
3. Render will auto-deploy (if enabled)
4. Monitor logs for successful deployment

### Adding New Features:
1. Test locally first
2. Update requirements.txt if needed
3. Deploy and verify functionality

## 🛡️ Security Considerations

### Data Privacy:
- **No Patient Data**: Ensure no patient information in repository
- **Local Processing**: PDF processing happens on Render servers
- **No External APIs**: No data sent to external services (except embedding model download)

### Access Control:
- Consider adding authentication for production use
- Use environment variables for sensitive configurations
- Regularly update dependencies for security patches

## 📞 Support

### Render.com Support:
- Documentation: https://render.com/docs
- Community Forum: https://community.render.com
- Support Email: support@render.com

### Application Issues:
- Check application logs in Render dashboard
- Test locally to isolate deployment-specific issues
- Verify all files are properly committed to Git

## 💰 Cost Estimation

### Free Tier Limitations:
- 750 hours/month (sufficient for testing)
- App sleeps after 15 minutes of inactivity
- Limited compute resources

### Paid Plans:
- **Starter ($7/month)**: Better for production use
- **Standard ($25/month)**: Higher performance
- **Pro ($85/month)**: Maximum performance

## 🎉 Success Checklist

After deployment, verify:
- [ ] App loads without errors
- [ ] Quiz functionality works
- [ ] Fluid calculator operates correctly
- [ ] PDF search returns results
- [ ] Logo and branding appear correctly
- [ ] All navigation tabs function
- [ ] Fallback messages appear for LLM features

Your KKH Nursing Chatbot should now be successfully deployed on Render.com! 🏥✨
