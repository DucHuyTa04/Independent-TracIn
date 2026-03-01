#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="${1:-config.yaml}"
JOB_NAME="${2:-tracin-ghost}"

mkdir -p /scratch/leihong/slurm_logs

JOB_ID=$(
  sbatch \
    --job-name="$JOB_NAME" \
    -o /scratch/leihong/slurm_logs/${JOB_NAME}-%j.out \
    -e /scratch/leihong/slurm_logs/${JOB_NAME}-%j.err \
    --export=ALL,CONFIG_PATH_INNER="$CONFIG_PATH" \
    "$ROOT_DIR/slurm_run_container.sh" \
    | awk '{print $4}'
)

echo "Submitted job: $JOB_ID"
echo "Track: squeue -j $JOB_ID"
echo "Logs:  /scratch/leihong/slurm_logs/${JOB_NAME}-${JOB_ID}.out"
