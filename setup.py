"""
Setup script for the Tariff Rates RAG system
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required Python packages"""
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    )

def check_gemini_api_key():
    """Check if Google Gemini API key is configured"""
    print("🔑 Checking Google Gemini API key...")
    
    # Check environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Check .env file
    if not api_key:
        env_file = Path(".env")
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.getenv("GOOGLE_API_KEY")
            except ImportError:
                pass
    
    if not api_key:
        print("❌ GOOGLE_API_KEY is not set!")
        print("   Please set it using one of these methods:")
        print("   1. Export: export GOOGLE_API_KEY=your_key_here")
        print("   2. Create .env file: echo 'GOOGLE_API_KEY=your_key_here' > .env")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        return False
    
    print(f"✅ Google Gemini API key is configured (length: {len(api_key)} characters)")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ["data", "vector_db", "models"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def test_system():
    """Test the system components"""
    print("\n🧪 Testing system components...")
    
    try:
        # Test data processing
        from data_processor import TariffDataProcessor
        processor = TariffDataProcessor()
        processor.load_data()
        print("✅ Data processor working")
        
        # Test vector store
        from vector_store import TariffVectorStore
        vector_store = TariffVectorStore()
        vector_store.initialize()
        print("✅ Vector store working")
        
        # Test LLM client
        from llm_client import BookLLMClient
        llm_client = BookLLMClient()
        if llm_client.setup():
            print("✅ LLM client working")
        else:
            print("❌ LLM client setup failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Tariff Rates RAG System")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Check Gemini API key
    if not check_gemini_api_key():
        return False
    
    # Test system
    if not test_system():
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nTo start the application, run:")
    print("  streamlit run app.py")
    print("\nTo test the system, run:")
    print("  python rag_pipeline.py")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
