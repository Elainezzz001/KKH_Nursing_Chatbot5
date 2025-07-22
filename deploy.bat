@echo off
REM Deployment script for KKH Nursing Chatbot to Fly.io (Windows)

echo 🚀 Starting deployment to Fly.io...

REM Check if flyctl is installed
flyctl version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ flyctl CLI is not installed. Please install it first:
    echo    https://fly.io/docs/hands-on/install-flyctl/
    exit /b 1
)

REM Check if logged in to Fly.io
flyctl auth whoami >nul 2>&1
if %errorlevel% neq 0 (
    echo 🔐 Please login to Fly.io first:
    echo    flyctl auth login
    exit /b 1
)

REM Check if API key is set
echo 🔑 Checking API key configuration...
if "%OPENROUTER_API_KEY%"=="" (
    echo ⚠️  OPENROUTER_API_KEY environment variable not set.
    set /p api_key=Enter your OpenRouter API key: 
    if "!api_key!"=="" (
        echo ❌ API key is required for deployment.
        exit /b 1
    )
    echo 📝 Setting API key secret...
    flyctl secrets set OPENROUTER_API_KEY="!api_key!"
) else (
    echo ✅ Using API key from environment variable.
    flyctl secrets set OPENROUTER_API_KEY="%OPENROUTER_API_KEY%"
)

REM Deploy the application
echo 🚀 Deploying application...
flyctl deploy

REM Check deployment status
if %errorlevel% equ 0 (
    echo ✅ Deployment successful!
    echo 🌐 Opening your application...
    flyctl open
) else (
    echo ❌ Deployment failed. Please check the logs:
    echo    flyctl logs
    exit /b 1
)

echo 🎉 Deployment complete!
echo.
echo Useful commands:
echo   flyctl logs     - View application logs
echo   flyctl status   - Check application status
echo   flyctl open     - Open application in browser
echo   flyctl ssh console - SSH into the running instance

pause
