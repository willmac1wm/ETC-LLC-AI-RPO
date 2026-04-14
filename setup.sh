#!/bin/bash

# ATC-Whisper Setup Script
# Creates a virtual environment and installs dependencies

echo "🚀 Starting ATC-Whisper setup..."

# Navigate to project directory
cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

echo "✅ Setup complete! You can now run the server with:"
echo "source venv/bin/activate && python3 code/atc_whisper_server.py"
