# Render Deployment Guide for KKH Nursing Chatbot

## 🚀 Quick Deploy to Render

Your KKH Nursing Chatbot is now ready for deployment to Render using the included `render.yaml` configuration.

### 📋 Prerequisites

1. **GitHub Repository:** ✅ Already set up at https://github.com/Elainezzz001/KKH_Nursing_Chatbot5.git
2. **Render Account:** Create a free account at [render.com](https://render.com)
3. **Cloud LM Studio:** ✅ Already configured at `35.247.130.124:1234`

### 🛠️ Deployment Steps

#### Option 1: Deploy via Render Dashboard (Recommended)

1. **Connect GitHub to Render:**
   - Go to [render.com](https://render.com) and sign up/login
   - Click "New +" → "Web Service"
   - Connect your GitHub account
   - Select the repository: `Elainezzz001/KKH_Nursing_Chatbot5`

2. **Configure Service:**
   - **Name:** `kkh-nursing-chatbot`
   - **Branch:** `main`
   - **Runtime:** `Python`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

3. **Environment Variables:**
   ```
   PYTHON_VERSION=3.10
   STREAMLIT_SERVER_HEADLESS=true
   STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
   ```

4. **Deploy:**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)
   - Your app will be available at: `https://kkh-nursing-chatbot.onrender.com`

#### Option 2: Deploy via render.yaml (Automatic)

1. **Fork and Deploy:**
   - Render will automatically detect the `render.yaml` file
   - All configurations are pre-set
   - Just click "Deploy" and wait for completion

### 🔧 Configuration Details

The `render.yaml` file includes:

```yaml
services:
  - type: web
    name: kkh-nursing-chatbot
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
    envVars:
      - key: PYTHON_VERSION
        value: "3.10"
      - key: STREAMLIT_SERVER_HEADLESS
        value: "true"
      - key: STREAMLIT_BROWSER_GATHER_USAGE_STATS
        value: "false"
```

### 📊 Expected Deployment Features

- ✅ **Free Tier:** No cost for basic usage
- ✅ **Auto-Deploy:** Automatic updates when you push to GitHub
- ✅ **HTTPS:** Secure connection included
- ✅ **Custom Domain:** Available if needed
- ✅ **Environment Variables:** Pre-configured for Streamlit

### 🔍 Testing Your Deployment

Once deployed, test these features:

1. **Semantic Search Chat:**
   - Try quick prompts: "Signs of dehydration", "Paediatric CPR steps"
   - Ask custom questions about nursing procedures

2. **Fluid Calculator:**
   - Enter patient weight and age
   - Test different clinical situations

3. **Knowledge Quiz:**
   - Start the 15-question quiz
   - Verify scoring and feedback work correctly

### 🐛 Troubleshooting

**Common Issues:**

1. **Build Failed:**
   - Check that all dependencies in `requirements.txt` are compatible
   - Verify Python version (3.10) is supported

2. **App Won't Start:**
   - Ensure the start command includes `--server.port $PORT --server.address 0.0.0.0`
   - Check logs in Render dashboard

3. **LM Studio Connection Issues:**
   - Verify cloud server at `35.247.130.124:1234` is running
   - Check firewall settings allow inbound connections

4. **PDF Loading Issues:**
   - Ensure `data/KKH Information file.pdf` is included in the repository
   - Check file permissions and accessibility

### 📞 Support

- **Render Documentation:** [docs.render.com](https://docs.render.com)
- **Streamlit on Render:** [docs.streamlit.io/deploy/render](https://docs.streamlit.io/deploy/render)
- **GitHub Repository:** https://github.com/Elainezzz001/KKH_Nursing_Chatbot5.git

### 🎉 Post-Deployment

After successful deployment:

1. **Share the URL** with your team
2. **Test all features** thoroughly
3. **Monitor usage** via Render dashboard
4. **Set up auto-deploy** for future updates

Your KKH Nursing Chatbot will be accessible worldwide at your Render URL! 🌍🏥
