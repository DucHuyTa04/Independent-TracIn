#!/usr/bin/env bash
#SBATCH --job-name=tracin-demo-full
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=03:00:00
#SBATCH -o /scratch/duchuy/slurm_logs/tracin-demo-full-%j.out
#SBATCH -e /scratch/duchuy/slurm_logs/tracin-demo-full-%j.err
#SBATCH --export=ALL
set -euo pipefail

# SLURM_SUBMIT_DIR is the cwd when sbatch was invoked (= project root).
# Fallback to dirname-based resolution for local runs.
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="/scratch/duchuy/venvs/tracenv/bin/python"
DATA_ROOT="/scratch/duchuy/tracin_benchmark_data"

module load StdEnv/2023 || true

cd "$ROOT_DIR"

echo "=========================================="
echo "  TracIn Demo — Full Training + Figures"
echo "  Job ID: ${SLURM_JOB_ID:-local}"
echo "  Node:   $(hostname)"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

# Phase 1: Pre-train all 3 models with full settings
echo ""
echo "Phase 1: Pre-training models..."
"$PYTHON" demos/pretrain_all.py \
    --device cuda \
    --data-root "$DATA_ROOT" \
    --cifar-epochs 10 \
    --gpt-epochs 15 \
    --vae-epochs 20 \
    --max-train 8000 \
    --n-train 8000 \
    --projection-dim 512

echo ""
echo "Phase 2: Generating attribution figures (headless)..."
"$PYTHON" demos/interactive_demo.py \
    --headless \
    --device cuda \
    --data-root "$DATA_ROOT" \
    --top-k 5 \
    --max-train 8000 \
    --n-train 8000 \
    --projection-dim 512

echo ""
echo "=========================================="
echo "  Done!  Figures saved under demos/outputs/"
echo "=========================================="
ls -la demos/outputs/*/attribution_*.png 2>/dev/null || echo "  (no figures found)"
