#!/bin/bash

# Resume Tailoring API Server Startup Script

echo "🚀 Starting Resume Tailoring API Server..."
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements_api.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install -r requirements_api.txt

# Create outputs directory
mkdir -p outputs

# Start the server
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "📱 Open http://localhost:8000 in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn api:app --reload --host 0.0.0.0 --port 8000
