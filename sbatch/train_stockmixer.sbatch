#!/bin/bash
#SBATCH --job-name=stockmixer_ndx
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --account=cuda
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn05,cn06
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/stockmixer_%j.out
#SBATCH --error=logs/stockmixer_%j.err

# ─────────────────────────────────────────────────────────────────────
# Train StockMixer on canonical NDX-100 panel.
#   seq_len=30  pred_len=5  batch=64  epochs=100  patience=15
# GPU: any modern GPU (K80 excluded — CC 3.7 unsupported by PyTorch 2+).
# Short time-limit + modest mem to catch backfill slots quickly.
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate

echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Started:    $(date)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="

export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"

python -u src/data/train_stockmixer.py \
    --panel data/processed/views_panel.parquet \
    --seq-len 30 \
    --pred-len 5 \
    --batch-size 64 \
    --epochs 100 \
    --lr 2e-4 \
    --patience 15 \
    --alpha 0.1 \
    --market-dim 2 \
    --seed 42 \
    --ckpt-dir checkpoints/canonical \
    --tag stockmixer_ndx100

echo "=========================================="
echo "Finished:   $(date)"
echo "=========================================="
