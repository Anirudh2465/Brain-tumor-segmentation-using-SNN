Replace `217.18.55.85` with `217.18.55.119` in the SSH/SCP commands. Everything else remains the same.

### Step 1 — Zip and transfer the dataset (Laptop)

```powershell
# 1. Zip the processed folder
Compress-Archive -Path "data\processed" -DestinationPath "processed.zip"

# 2. Transfer the zip to Jarvis Labs
scp -o StrictHostKeyChecking=no processed.zip root@217.18.55.119:/workspace/
```

### Step 2 — Connect to the machine (Laptop)

```powershell
ssh -o StrictHostKeyChecking=no root@217.18.55.119
```

### Step 3 — Setup the code and data (inside Jarvis)

```bash
cd /workspace

git clone -b feat/jarvis-final https://github.com/Anirudh2465/Brain-tumor-segmentation-using-SNN.git

cd Brain-tumor-segmentation-using-SNN

# Move the zip into the project and unpack it
mkdir -p data/
mv /workspace/processed.zip data/
cd data
unzip processed.zip
rm processed.zip
cd ..

# Install all dependencies
bash setup_jarvis.sh
```

### Step 4 — Start training (inside Jarvis)

For **2× A100s**:

```bash
python scripts/run_full_pipeline.py --skip_preprocess --epochs 50 --batch_size 32
```

For **2× L4s**:

```bash
python scripts/run_full_pipeline.py --skip_preprocess --epochs 50 --batch_size 16
```

The new SSH endpoint is therefore:

```text
root@217.18.55.119
```

No other command changes are needed.
