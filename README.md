# bookbot 📚
bookbot

A powerful Retrieval-Augmented Generation (RAG) system for querying book summaries and information using **Amazon Bedrock** (Claude) and **ChromaDB** vector database.

## ✨ Features

- 🔍 **Semantic Search**: Find relevant book information using natural language queries
- 🤖 **Amazon Bedrock Integration**: Powered by Anthropic Claude models via AWS Bedrock
- 📊 **Vector Database**: ChromaDB for fast similarity search and document retrieval
- 💬 **Interactive Chat Interface**: User-friendly Streamlit web interface
- 🎯 **Smart RAG Mode**: Automatically retrieves relevant book summaries and generates answers
- 🎛️ **Temperature Control**: Adjustable response creativity
- 🔐 **Secure API Key Management**: Environment variable and .env file support
- 🚀 **FastAPI REST API**: Programmatic access to the RAG system

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CSV Data +     │───▶│  Data Processor  │───▶│  Vector Store │
│ transcripts     │    |   (Chunking &    │    │    (ChromaDB)   │
│ (Book Summaries)│    │   Embeddings)    │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐              │
│  Streamlit UI   │◀───│   RAG Pipeline   │◀────────────┘
│ (Chat Interface)│    │ (Retrieval +    │
│                 │    │  Generation)     │
└─────────────────┘    └──────────────────┘
                                │
                       ┌──────────────────┐
                       │  Amazon Bedrock  │
                       │  (Claude Model) │
                       |  OR other model  |
                       └──────────────────┘
```

## 📋 Prerequisites

1. **Python 3.9+** (tested with Python 3.9)
2. **Amazon Bedrock API Key** - Get from AWS Bedrock Console
3. **CSV Data File** - Place your book summaries CSV file in the `data/` directory

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd bookbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Bedrock API key**
   
   Create a `.env` file in the project root:
   ```bash
   # Create .env file
   cat > .env << EOF
   # Amazon Bedrock API Key
   BEDROCK_API_KEY=your_bedrock_api_key_here
   
   # AWS Region for Bedrock
   AWS_REGION=us-east-1
   EOF
   
   # Protect the file
   chmod 600 .env
   ```

4. **Add your book summaries data**
   
   Place your CSV file in the `data/` directory:
   ```bash
   # Your CSV should be named: exported_books_db_merged.csv
   # Or update config.py to point to your file
   ```

5. **Start the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** to `http://localhost:8501`

### Option 2: Manual Setup

1. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Set your Bedrock API key**
   
   Create a `.env` file:
   ```bash
   echo "BEDROCK_API_KEY=your_api_key_here" > .env
   echo "AWS_REGION=us-east-1" >> .env
   chmod 600 .env
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## ☁️ AWS EC2 Deployment

Deploy this application on AWS EC2 instance from scratch.

### Quick Start

1. **Launch EC2 Instance** (Ubuntu 22.04, t3.small recommended)
2. **Connect via SSH**
3. **Run setup script:**
   ```bash
   # Upload ec2-setup.sh to your EC2 instance, then:
   chmod +x ec2-setup.sh
   ./ec2-setup.sh
   ```

### Complete Guide

See **[EC2_DEPLOYMENT_GUIDE.md](EC2_DEPLOYMENT_GUIDE.md)** for detailed step-by-step instructions covering:
- Launching EC2 instance
- Installing dependencies
- Configuring the application
- Setting up auto-start service
- Troubleshooting
- Optional: Nginx reverse proxy

## 📁 Project Structure

```
bookbot/
├── app.py                 # Streamlit web interface (main chat UI)
├── api.py                 # FastAPI REST API endpoint
├── config.py              # Configuration settings
├── data_processor.py      # CSV processing and chunking
├── vector_store.py        # ChromaDB vector operations
├── llm_client.py          # Amazon Bedrock LLM integration
├── rag_pipeline.py        # Main RAG pipeline orchestrator
├── interactive_chat.py    # Interactive command-line chat
├── setup.py               # Automated setup script
├── start.sh               # Startup script
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .env                   # Environment variables (create this)
├── data/                  # CSV data files
│   └── exported_books_db_merged.csv
├── vector_db/            # ChromaDB persistent storage
└── models/                # Cached embedding models
```

## 💻 Usage

### Web Interface (Streamlit)

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Features available:**
   - **Chat Interface**: Ask questions about books in natural language
   - **Temperature Control**: Adjust response creativity (0.0-1.0)
   - **Clear Chat**: Reset conversation history
   - **Send transcript**: Send transcript of chat to user

3. **Example Queries:**
   - "What books are about marketing?"
   - "Tell me about the Classical Marketing Book"
   - "Find books on business communication"
   - "What are books about AI and machine learning?"
   - "Summarize the main themes in marketing books"

### REST API (FastAPI)

1. **Start the API server**
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

2. **API Endpoints:**

   **POST /chat** - Query the RAG system
   ```bash
   curl -X POST "http://localhost:8000/chat" \
        -H "Content-Type: application/json" \
        -d '{"question": "What books are about marketing?", "language": "en"}'
   ```
   
   Response:
   ```json
   {
     "answer": "...",
     "processing_time": 1.23,
     "sources": [...]
   }
   ```

### Interactive Command-Line Chat

```bash
python interactive_chat.py
```

## ⚙️ Configuration

Edit `config.py` or use environment variables to customize:

- **LLM Model**: Default is `anthropic.claude-3-sonnet-20240229-v1:0`
  - Supports Claude models via Amazon Bedrock
  - Can be changed via `BEDROCK_MODEL` environment variable

- **Embedding Model**: Default is `all-MiniLM-L6-v2` (Sentence Transformers)

- **RAG Settings**:
  - `CHUNK_SIZE`: 1000 (characters per chunk)
  - `CHUNK_OVERLAP`: 200 (overlap between chunks)
  - `TOP_K_RESULTS`: 5 (number of documents to retrieve)

- **Data Files**: Configure paths in `config.py` or place CSV files in `data/` directory

## 📊 Data Format

Your CSV file should have these columns:

**Required Columns:**
- `id`: Unique book identifier
- `book_title`: Title of the book
- `book_category`: Category/genre of the book
- `book_description`: Description of the book
- `book_blob_notes`: Detailed notes/summary (main content)

**Optional Columns:**
- `book_subtitle`: Subtitle of the book
- `book_pubyear`: Publication year
- `book_excerpt2`, `book_excerpt3`: Additional excerpts
- `goodreads_url`, `book_amazon_us`: Links to book pages
- `podcast_shownote_url`: Related podcast links

## 🔧 Environment Variables

The application supports configuration via environment variables or `.env` file:

### Using .env file (Recommended)

Create a `.env` file in the project root:

```bash
# Amazon Bedrock API Key (required)
BEDROCK_API_KEY=your_bedrock_api_key_here

# AWS Region (required)
AWS_REGION=us-east-1

# Bedrock Model (optional - defaults to anthropic.claude-3-sonnet-20240229-v1:0)
# BEDROCK_MODEL=anthropic.claude-3-sonnet-20240229-v1:0

# Alternative: Use standard AWS credentials instead of API key
# AWS_ACCESS_KEY_ID=your_access_key_id_here
# AWS_SECRET_ACCESS_KEY=your_secret_access_key_here
```

### Using Environment Variables

```bash
export BEDROCK_API_KEY=your_bedrock_api_key_here
export AWS_REGION=us-east-1
```

## 🔑 Getting Your Bedrock API Key

1. **Go to AWS Bedrock Console**: https://console.aws.amazon.com/bedrock
2. **Navigate to API Keys** in the left navigation pane
3. **Generate API Key**:
   - Choose "Generate long-term API keys" for persistent access
   - Set expiration period if needed
   - Click "Generate"
4. **Copy the API key** (starts with `ABSK...`)
5. **Add to your `.env` file** as shown above

## 🛠️ Troubleshooting

### Bedrock API Issues

**Check if API key is set:**
```bash
python -c "from config import BEDROCK_API_KEY; print('✅ Key set' if BEDROCK_API_KEY else '❌ Key not set')"
```

**Verify API key in .env file:**
```bash
cat .env
```

**Common issues:**
- **Invalid model identifier**: The default model `anthropic.claude-3-sonnet-20240229-v1:0` should work. If you get model errors, try:
  - `anthropic.claude-3-haiku-20240307-v1:0` (faster, cheaper)
  - `anthropic.claude-3-opus-20240229-v1:0` (more capable)
- **UnrecognizedClientException**: Your API key format might be incorrect. Ensure it starts with `ABSK` and is the full key from AWS Console
- **Access denied**: Check that your API key has permissions to access Bedrock models
- **Network issues**: Verify internet connectivity and AWS region availability

### Vector Database Issues

**Clear and rebuild the database:**
```bash
python -c "
from vector_store import BookVectorStore
vs = BookVectorStore()
vs.initialize()
vs.clear_collection()
vs.populate_database()
"
```

**Check database status:**
```bash
python -c "
from vector_store import BookVectorStore
vs = BookVectorStore()
vs.initialize()
info = vs.get_collection_info()
print(f'Documents: {info.get(\"document_count\", 0)}')
"
```

### Authentication Methods

The system supports two authentication methods:

1. **Bedrock API Key (Recommended)**: 
   - Format: `ABSK...` (base64 encoded)
   - Set via `BEDROCK_API_KEY` in `.env`
   - Uses direct HTTP API calls

2. **Standard AWS Credentials**:
   - Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in `.env`
   - Uses boto3 with standard AWS authentication
   - Requires IAM permissions for Bedrock

### Import Errors

**Ensure all dependencies are installed:**
```bash
pip install -r requirements.txt
```

**Check Python version:**
```bash
python --version  # Should be 3.9+
```

## 📦 Dependencies

Key dependencies:
- `streamlit>=1.28.0` - Web interface
- `chromadb>=0.5.0` - Vector database
- `sentence-transformers>=2.2.0` - Embeddings
- `boto3>=1.30.0` - AWS SDK (for Bedrock)
- `pandas>=2.0.0` - Data processing
- `fastapi>=0.110.0` - REST API
- `uvicorn[standard]>=0.27.0` - ASGI server
- `python-dotenv>=1.0.0` - Environment variables
- `requests>=2.31.0` - HTTP requests for Bedrock API

See `requirements.txt` for the complete list.

## 🚀 Performance Tips

1. **GPU Acceleration**: Install CUDA-compatible PyTorch for faster embeddings
2. **Chunk Size**: Smaller chunks (500-800) for precise queries, larger (1000-1500) for general topics
3. **Model Selection**: Claude 3 Haiku is faster and cheaper, Claude 3 Sonnet balances speed and quality
4. **Memory**: 8GB+ RAM recommended for smooth operation
5. **Caching**: Vector database persists between runs for faster startup
6. **Temperature**: Lower values (0.1-0.3) for factual queries, higher (0.7-1.0) for creative responses

## 🔒 Security Notes

- Never commit your `.env` file or API keys to version control
- Use `chmod 600 .env` to protect your API key file
- The `.env` file is automatically added to `.gitignore`
- Consider using AWS IAM roles in production environments
- Review AWS Bedrock pricing and quota limits
- Rotate API keys regularly

## 🧪 Testing

Test individual components:

```bash
# Test LLM client
python -c "from llm_client import BookLLMClient; client = BookLLMClient(); print('✅ LLM OK' if client.setup() else '❌ LLM Failed')"

# Test vector store
python -c "from vector_store import BookVectorStore; vs = BookVectorStore(); vs.initialize(); print('✅ Vector Store OK')"

# Test RAG pipeline
python -c "from rag_pipeline import BookRAGPipeline; pipeline = BookRAGPipeline(); print('✅ Pipeline OK' if pipeline.initialize() else '❌ Pipeline Failed')"

# Run full system test
python test_system.py
```

## 📝 Example Usage

### Basic Query
```python
from rag_pipeline import BookRAGPipeline

pipeline = BookRAGPipeline()
pipeline.initialize()

result = pipeline.query("What books are about marketing?")
print(result['answer'])
```

### Query with Filters
```python
result = pipeline.query_with_filters(
    question="Find books on business",
    book_category="Business & Economics",
    min_year=2020
)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Amazon Bedrock** for LLM capabilities
- **Anthropic** for Claude models
- **ChromaDB** for vector database
- **Streamlit** for the web interface

