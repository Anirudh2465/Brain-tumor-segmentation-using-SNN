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
echo "Generating data splits from uploaded processed data..."
python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'src')
from spiking_useg.data.splits import discover_case_ids, generate_splits

processed_dir = Path('data/processed')
splits_dir = Path('data/splits')
if processed_dir.exists():
    ids = discover_case_ids(processed_dir)
    if ids:
        generate_splits(ids, n_folds=1, splits_dir=splits_dir)
        print(f'Generated splits for {len(ids)} cases.')
    else:
        print('No processed cases found. Training will fail.')
"

echo "Which view would you like to train? (Enter: axial, coronal, sagittal, or all)"
read -p "> " view_choice
view_choice=$(echo "$view_choice" | tr -d '\r')

if [ "$view_choice" = "all" ]; then
    echo "Starting training for ALL views..."
    python scripts/run_full_pipeline.py --skip_preprocess --epochs 20 --batch_size 16
elif [ "$view_choice" = "axial" ] || [ "$view_choice" = "coronal" ] || [ "$view_choice" = "sagittal" ]; then
    echo "Starting training for $view_choice view only..."
    python scripts/run_full_pipeline.py --skip_preprocess --epochs 20 --batch_size 16 --views "$view_choice"
else
    echo "Invalid choice: '$view_choice'. Please run the script again."
    exit 1
fi
