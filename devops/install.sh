#!/bin/bash
set -e

echo "Installing dependencies..."
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu121

echo "Installing FAISS..."
# Try system package first, fallback to pip
if command -v apt-get &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y libfaiss-dev
fi
pip install faiss-cpu==1.7.4

echo "Installing project..."
pip install -e .

echo "Installation complete!"
