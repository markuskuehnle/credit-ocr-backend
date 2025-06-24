#!/bin/bash
"""
Startup script for the OCR System for Credit Evaluation

Usage:
    ./start.sh [port]

Examples:
    ./start.sh          # Start on default port 8000
    ./start.sh 8080     # Start on port 8080
"""

# Set default port
PORT=${1:-8000}

echo "Starting Credit OCR Demo Backend API on port $PORT"
echo "Environment variables:"
echo "  HOST=0.0.0.0"
echo "  PORT=$PORT"
echo "  RELOAD=false"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Start the API server
python run.py 