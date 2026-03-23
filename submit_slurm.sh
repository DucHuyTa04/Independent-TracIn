#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="${1:-config.yaml}"
JOB_NAME="${2:-tracin-ghost}"
MODE="${3:-index}"
INPUT_PATH="${4:-${INPUT_PATH:-}}"
# For mode=full, run full pipeline (train->index->query) in one job

mkdir -p /scratch/duchuy/slurm_logs

JOB_ID=$(
  sbatch \
    --job-name="$JOB_NAME" \
    -o /scratch/duchuy/slurm_logs/${JOB_NAME}-%j.out \
    -e /scratch/duchuy/slurm_logs/${JOB_NAME}-%j.err \
    --export=ALL,CONFIG_PATH_INNER="$CONFIG_PATH",MODE="$MODE",INPUT_PATH="$INPUT_PATH" \
    "$ROOT_DIR/slurm_run_container.sh" \
    | awk '{print $4}'
)

echo "Submitted job: $JOB_ID"
echo "Track: squeue -j $JOB_ID"
echo "Logs:  /scratch/duchuy/slurm_logs/${JOB_NAME}-${JOB_ID}.out"
