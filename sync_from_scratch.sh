#!/usr/bin/env bash
# Pull Slurm job outputs from $SCRATCH_BASE into this repo for inspection.
#
# Usage:
#   ./sync_from_scratch.sh [job_id | all]
#
# Syncs per job:
#   - outputs/   (FAISS, checkpoints, benchmarks, etc.)
#   - logs/
#   - data/      (e.g. downloaded MNIST/CIFAR under the job result dir)
#
# Environment:
#   SCRATCH_BASE  top-level scratch dir (default: /scratch/$USER)
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_FILTER="${1:-all}"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/${USER:-$LOGNAME}}"

mkdir -p "$ROOT_DIR/outputs/runs" "$ROOT_DIR/logs/runs" "$ROOT_DIR/logs/slurm" "$ROOT_DIR/data/runs"

sync_one_job() {
	local job_id="$1"
	local scratch_job_dir="${SCRATCH_BASE}/tracin_ghost_tool_results/${job_id}"

	if [[ ! -d "$scratch_job_dir" ]]; then
		echo "Skip missing $scratch_job_dir"
		return 0
	fi

	if [[ -d "${scratch_job_dir}/outputs" ]]; then
		mkdir -p "$ROOT_DIR/outputs/runs/${job_id}"
		rsync -a "${scratch_job_dir}/outputs/" "$ROOT_DIR/outputs/runs/${job_id}/"
	fi

	if [[ -d "${scratch_job_dir}/logs" ]]; then
		mkdir -p "$ROOT_DIR/logs/runs/${job_id}"
		rsync -a "${scratch_job_dir}/logs/" "$ROOT_DIR/logs/runs/${job_id}/"
	fi

	if [[ -d "${scratch_job_dir}/data" ]]; then
		mkdir -p "$ROOT_DIR/data/runs/${job_id}"
		rsync -a "${scratch_job_dir}/data/" "$ROOT_DIR/data/runs/${job_id}/"
	fi

	if [[ -f "${scratch_job_dir}/outputs/attribution_results.json" ]]; then
		mkdir -p "$ROOT_DIR/outputs"
		cp -f "${scratch_job_dir}/outputs/attribution_results.json" "$ROOT_DIR/outputs/attribution_results.json"
	fi

	if [[ -f "${scratch_job_dir}/outputs/faiss_index" ]]; then
		mkdir -p "$ROOT_DIR/outputs"
		cp -f "${scratch_job_dir}/outputs/faiss_index" "$ROOT_DIR/outputs/faiss_index" 2>/dev/null || true
	fi

	if [[ -f "${scratch_job_dir}/outputs/faiss_metadata.json" ]]; then
		mkdir -p "$ROOT_DIR/outputs"
		cp -f "${scratch_job_dir}/outputs/faiss_metadata.json" "$ROOT_DIR/outputs/faiss_metadata.json"
	fi
}

if [[ "$JOB_FILTER" == "all" ]]; then
	if [[ -d "${SCRATCH_BASE}/tracin_ghost_tool_results" ]]; then
		for job_id in $(ls -1 "${SCRATCH_BASE}/tracin_ghost_tool_results" 2>/dev/null | sort -n); do
			sync_one_job "$job_id"
		done
	fi
else
	sync_one_job "$JOB_FILTER"
fi

rsync -a "${SCRATCH_BASE}/slurm_logs/" "$ROOT_DIR/logs/slurm/" 2>/dev/null || true

echo "Sync completed."
