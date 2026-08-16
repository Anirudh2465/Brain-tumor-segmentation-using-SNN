#!/bin/bash
# setup_jarvis.sh - Environment setup script for Jarvis Labs

echo "================================================="
echo "   Setting up SpikingUSegNet on Jarvis Labs      "
echo "================================================="

# 1. Install pip dependencies
echo "[1/2] Installing requirements..."
pip install -r requirements.txt

# 2. Prepare Data directory structure
echo "[2/2] Preparing data directories..."
mkdir -p data/processed
mkdir -p experiments

echo "================================================="
echo "Setup Complete!"
echo "Next Steps:"
echo "1. Upload your processed dataset to data/processed/"
echo "   (Drag and drop the 'processed' folder into the JupyterLab file browser)"
echo "2. Run training with a massive batch size (e.g. 16 or 32):"
echo "   python scripts/run_full_pipeline.py --skip_preprocess --views axial coronal sagittal --folds 0 --batch_size 16"
echo "================================================="
