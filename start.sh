#!/bin/bash
set -e

echo "🚀 Starting BookBot RAG Assistant"
echo "=============================================="

# Check if GOOGLE_API_KEY is set
if [ -z "$GOOGLE_API_KEY" ] && [ ! -f .env ]; then
    echo "⚠️  Warning: GOOGLE_API_KEY not set and .env file not found"
    echo "   Please set your API key:"
    echo "   export GOOGLE_API_KEY=your_key"
    echo "   Or create a .env file with: GOOGLE_API_KEY=your_key"
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Start Streamlit
echo "🌐 Launching Streamlit on port ${STREAMLIT_SERVER_PORT:-8501}..."
exec streamlit run app.py \
  --server.port="${STREAMLIT_SERVER_PORT:-8501}" \
  --server.address=0.0.0.0
