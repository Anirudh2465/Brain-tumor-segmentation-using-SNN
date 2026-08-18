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
echo "================================================="
echo ""
echo "Please ensure your 'processed' dataset is extracted to data/processed before continuing."
echo ""
echo "Which view would you like to train? (Enter: axial, coronal, sagittal, or all)"
read -p "> " view_choice

if [ "$view_choice" == "all" ]; then
    echo "Starting training for ALL views..."
    python scripts/run_full_pipeline.py --skip_preprocess --epochs 50 --batch_size 32
elif [[ "$view_choice" =~ ^(axial|coronal|sagittal)$ ]]; then
    echo "Starting training for $view_choice view only..."
    python scripts/run_full_pipeline.py --skip_preprocess --epochs 50 --batch_size 32 --views $view_choice
else
    echo "Invalid choice. Please run the script again or manually run run_full_pipeline.py"
    exit 1
fi
