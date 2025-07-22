#!/usr/bin/env python3
"""
Setup script for KKH Nursing Chatbot development environment
"""

import os
import sys
import subprocess
import venv
import shutil

def create_virtual_environment():
    """Create a virtual environment if it doesn't exist"""
    venv_path = "venv"
    
    if os.path.exists(venv_path):
        print(f"✅ Virtual environment already exists at {venv_path}")
        return True
    
    print(f"📦 Creating virtual environment at {venv_path}...")
    try:
        venv.create(venv_path, with_pip=True)
        print("✅ Virtual environment created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def install_dependencies():
    """Install project dependencies"""
    print("📥 Installing dependencies...")
    
    # Determine the correct pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = os.path.join("venv", "Scripts", "pip.exe")
    else:  # Unix-like
        pip_path = os.path.join("venv", "bin", "pip")
    
    if not os.path.exists(pip_path):
        print(f"❌ Pip not found at {pip_path}")
        return False
    
    try:
        # Upgrade pip first
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_secrets():
    """Setup secrets file from template"""
    secrets_dir = ".streamlit"
    secrets_file = os.path.join(secrets_dir, "secrets.toml")
    template_file = "secrets.toml.example"
    
    # Create .streamlit directory if it doesn't exist
    if not os.path.exists(secrets_dir):
        os.makedirs(secrets_dir)
        print(f"✅ Created {secrets_dir} directory")
    
    # Copy template if secrets file doesn't exist
    if not os.path.exists(secrets_file):
        if os.path.exists(template_file):
            shutil.copy(template_file, secrets_file)
            print(f"✅ Created {secrets_file} from template")
            print("🔑 Please edit .streamlit/secrets.toml with your OpenRouter API key")
        else:
            print(f"⚠️  Template file {template_file} not found")
            return False
    else:
        print(f"✅ Secrets file already exists at {secrets_file}")
    
    return True

def main():
    """Main setup function"""
    print("🏥 KKH Nursing Chatbot - Development Environment Setup")
    print("=" * 55)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ required, but {sys.version} is installed")
        return 1
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} is compatible")
    
    # Setup steps
    steps = [
        ("Creating virtual environment", create_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Setting up secrets", setup_secrets),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f"❌ Failed: {step_name}")
            return 1
    
    print("\n" + "=" * 55)
    print("🎉 Setup complete!")
    print("\n📝 Next steps:")
    print("1. Edit .streamlit/secrets.toml with your OpenRouter API key")
    print("2. Run the application:")
    
    if os.name == 'nt':  # Windows
        print("   run_local.bat")
    else:  # Unix-like
        print("   ./run_local.sh")
    
    print("3. Or manually activate the environment and run:")
    
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix-like
        print("   source venv/bin/activate")
    
    print("   streamlit run app.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
