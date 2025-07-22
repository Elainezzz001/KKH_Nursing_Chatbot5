#!/bin/bash

# Deployment script for KKH Nursing Chatbot to Fly.io

echo "🚀 Starting deployment to Fly.io..."

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "❌ flyctl CLI is not installed. Please install it first:"
    echo "   https://fly.io/docs/hands-on/install-flyctl/"
    exit 1
fi

# Check if logged in to Fly.io
if ! flyctl auth whoami &> /dev/null; then
    echo "🔐 Please login to Fly.io first:"
    echo "   flyctl auth login"
    exit 1
fi

# Check if API key is set
echo "🔑 Checking API key configuration..."
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠️  OPENROUTER_API_KEY environment variable not set."
    read -p "Enter your OpenRouter API key: " api_key
    if [ -z "$api_key" ]; then
        echo "❌ API key is required for deployment."
        exit 1
    fi
    echo "📝 Setting API key secret..."
    flyctl secrets set OPENROUTER_API_KEY="$api_key"
else
    echo "✅ Using API key from environment variable."
    flyctl secrets set OPENROUTER_API_KEY="$OPENROUTER_API_KEY"
fi

# Deploy the application
echo "🚀 Deploying application..."
flyctl deploy

# Check deployment status
if [ $? -eq 0 ]; then
    echo "✅ Deployment successful!"
    echo "🌐 Opening your application..."
    flyctl open
else
    echo "❌ Deployment failed. Please check the logs:"
    echo "   flyctl logs"
    exit 1
fi

echo "🎉 Deployment complete!"
echo ""
echo "Useful commands:"
echo "  flyctl logs     - View application logs"
echo "  flyctl status   - Check application status"
echo "  flyctl open     - Open application in browser"
echo "  flyctl ssh console - SSH into the running instance"
