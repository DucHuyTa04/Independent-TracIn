#!/usr/bin/env bash
# Quick training entry point for Trillium / Compute Canada.
#
# Local (GPU login smoke test; uses native modules + conda env — see run_container.sh):
#   ./run_training.sh <mnist|cifar10_cnn|synth_regression> [train.py args...]
#
# Slurm GPU job (submit from GPU login; see submit_slurm.sh):
#   ./run_training.sh <model> --slurm [train.py args...]
#
# Environment: same as run_container.sh (TRAINING_ENV_ROOT, TRAIN_PYTHON, SCRATCH, …).
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:?Usage: $0 <mnist|cifar10_cnn|synth_regression> [--slurm] [train.py args...]}"
shift

if [[ "${1:-}" == "--slurm" ]]; then
	shift
	exec "$ROOT_DIR/submit_slurm.sh" train "$MODEL" "$@"
fi

exec "$ROOT_DIR/run_container.sh" native train "$MODEL" "$@"
