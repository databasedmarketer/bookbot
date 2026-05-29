"""
Configuration settings for the RAG LLM system
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import boto3

# Load environment variables from a .env file if present
load_dotenv()

# --- Project paths ---
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
try:
    DATA_DIR.mkdir(exist_ok=True)
    VECTOR_DB_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
except PermissionError:
    pass  # Directories may already exist or be created by user

# --- Data source settings (S3) ---
S3_BUCKET = "s3readingroom"   # 🔹 Replace with appropriate S3 bucket name

# Book summaries CSV file
BOOKS_CSV_FILE = DATA_DIR / "exported_books_db_merged.csv"

# --- Model settings ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Amazon Bedrock settings ---
# Option 1: Use Bedrock API key (ABSK format) - will be decoded
# IMPORTANT: Never hardcode API keys! Always use environment variables or .env file
BEDROCK_API_KEY = os.getenv("BEDROCK_API_KEY", "")

# Option 2: Use standard AWS credentials (preferred if available)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# For API keys, use: anthropic.claude-3-sonnet-20240229-v1:0
# For standard AWS credentials, can use: anthropic.claude-3-5-sonnet-20241022-v2:0
# BEDROCK_MODEL = os.getenv("BEDROCK_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0")

BEDROCK_MODEL = os.getenv("BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-6")
# Alternative models to try if default fails
BEDROCK_MODEL_ALTERNATIVES = [
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
]
BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1")

# --- Gemini settings (deprecated, kept for backward compatibility) ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- ChromaDB settings ---
COLLECTION_NAME_BOOKS = "csv_txt_embeddings"

# --- RAG settings ---
TOP_K_RESULTS = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Streamlit settings ---
PAGE_TITLE = "Glenn's Reading Room"
PAGE_ICON = "📚"
