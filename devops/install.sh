#!/bin/bash
set -e

echo "Installing dependencies..."
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu121

echo "Installing FAISS..."
pip install faiss-cpu==1.7.4

echo "Installing project..."
pip install -e .

echo "Installation complete!"
