#!/bin/bash
set -e

echo "Installing compatible numpy..."
pip install "numpy>=1.21.0,<2.0.0"

echo "Installing PyTorch dependencies..."
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu121

echo "Installing other dependencies..."
pip install -r requirements.txt

echo "Installing FAISS..."
pip install faiss-cpu==1.12.0

echo "Installing project..."
pip install -e .

echo "Installation complete!"
