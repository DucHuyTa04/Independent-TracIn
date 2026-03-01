#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_FILTER="${1:-all}"

mkdir -p "$ROOT_DIR/outputs/runs" "$ROOT_DIR/logs/runs" "$ROOT_DIR/logs/slurm" "$ROOT_DIR/outputs/results"

sync_one_job() {
  local job_id="$1"
  local scratch_job_dir="/scratch/leihong/tracin_ghost_tool_results/${job_id}"

  if [[ -d "${scratch_job_dir}/outputs" ]]; then
    mkdir -p "$ROOT_DIR/outputs/runs/${job_id}"
    rsync -a "${scratch_job_dir}/outputs/" "$ROOT_DIR/outputs/runs/${job_id}/"
  fi

  if [[ -d "${scratch_job_dir}/logs" ]]; then
    mkdir -p "$ROOT_DIR/logs/runs/${job_id}"
    rsync -a "${scratch_job_dir}/logs/" "$ROOT_DIR/logs/runs/${job_id}/"
  fi

  if [[ -f "${scratch_job_dir}/outputs/results/topk.csv" ]]; then
    cp -f "${scratch_job_dir}/outputs/results/topk.csv" "$ROOT_DIR/outputs/results/topk.csv"
  fi

  if [[ -f "${scratch_job_dir}/outputs/results/topk.json" ]]; then
    cp -f "${scratch_job_dir}/outputs/results/topk.json" "$ROOT_DIR/outputs/results/topk.json"
  fi
}

if [[ "$JOB_FILTER" == "all" ]]; then
  for job_id in $(ls -1 /scratch/leihong/tracin_ghost_tool_results 2>/dev/null | sort -n); do
    sync_one_job "$job_id"
  done
else
  sync_one_job "$JOB_FILTER"
fi

rsync -a /scratch/leihong/slurm_logs/tracin-ghost-* "$ROOT_DIR/logs/slurm/" 2>/dev/null || true

echo "Sync completed."
