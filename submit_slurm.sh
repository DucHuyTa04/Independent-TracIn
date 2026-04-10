#!/usr/bin/env bash
# Submit Slurm jobs for TracIn Ghost (main.py), testModels training, or GPU benchmarks.
#
# Usage — index / query / full (CPU job; submit from CPU login on Trillium):
#   ./submit_slurm.sh [config.yaml] [job_name] [mode] [input_path]
#
# Usage — training (GPU job; submit from GPU login — adds --gpus-per-node=1 per Trillium):
#   ./submit_slurm.sh train <mnist|cifar10_cnn|synth_regression> [job_name] [train.py args...]
#   If you omit job_name, it defaults to tracin-train-<model>.
#
# Usage — benchmarks (GPU; runs benchmarks/run_all.py via slurm_run_container.sh):
#   ./submit_slurm.sh benchmark <smoke|full> [optional_job_name_prefix]
#   smoke: small n/epochs (~2h default walltime); full: default heavy run_benchmark settings (~72h).
#
#   ./submit_slurm.sh benchmark-chain   # submit smoke, then full with afterok:smoke
#
# Optional env:
#   SLURM_LOG_DIR     stdout/stderr directory (default: /scratch/$USER/slurm_logs)
#   TRAIN_DATA_ROOT   export before sbatch for custom --data-root (benchmarks download MNIST/CIFAR here)
#   TRAIN_OUTPUT_DIR  rare override for checkpoint dir
#   TRAINING_EXTRA_SBATCH  optional extra sbatch flags (unquoted expansion), e.g. --account=...
#   BENCH_TIME_SMOKE  walltime for benchmark smoke (default 01:00:00)
#   BENCH_TIME_FULL   walltime for benchmark full (default 06:00:00)
#   TRACIN_EXTRA_MODULES  CUDA modules after StdEnv/2023 (default cudacore/12.6.3); set from
#                         `module spider cuda` on your cluster, or "none" if not needed.
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/scratch/${USER:-$LOGNAME}/slurm_logs}"
mkdir -p "$SLURM_LOG_DIR"

SUBMIT_KIND="index"

# Shorter defaults fit typical partition maxima; override on your site (e.g. BENCH_TIME_FULL=72:00:00).
BENCH_TIME_SMOKE="${BENCH_TIME_SMOKE:-01:00:00}"
BENCH_TIME_FULL="${BENCH_TIME_FULL:-06:00:00}"

_submit_bench_gpu() {
	local kind="$1"
	local dep="${2:-}"
	local name="${3:-tracin-bench-${kind}}"
	local walltime="$BENCH_TIME_SMOKE"
	[[ "$kind" == "full" ]] && walltime="$BENCH_TIME_FULL"
	local depargs=()
	[[ -n "$dep" ]] && depargs=(--dependency=afterok:"$dep")
	# shellcheck disable=SC2086
	sbatch "${depargs[@]}" \
		--job-name="$name" \
		--partition=compute \
		--nodes=1 \
		--ntasks=1 \
		--gpus-per-node=1 \
		--cpus-per-task=24 \
		--time="$walltime" \
		-o "$SLURM_LOG_DIR/${name}-%j.out" \
		-e "$SLURM_LOG_DIR/${name}-%j.err" \
		--export=ALL \
		${TRAINING_EXTRA_SBATCH:-} \
		"$ROOT_DIR/slurm_run_container.sh" benchmark "$kind" |
		awk '{print $4}'
}

if [[ "${1:-}" == "train" ]]; then
	SUBMIT_KIND="train"
	shift
	MODEL="${1:?train requires model: mnist | cifar10_cnn | synth_regression}"
	shift
	if [[ "${1:-}" != "" && "${1:-}" != -* ]]; then
		JOB_NAME="$1"
		shift
	else
		JOB_NAME="tracin-train-${MODEL}"
	fi
	# Trillium GPU: 1 GPU = quarter node (~24 cores). Partition name is still "compute" on GPU subcluster.
	# See https://docs.alliancecan.ca/wiki/Trillium_Quickstart#Submitting_jobs_for_the_GPU_subcluster
	# shellcheck disable=SC2086
	JOB_ID=$(
		sbatch \
			--job-name="$JOB_NAME" \
			--partition=compute \
			--nodes=1 \
			--ntasks=1 \
			--gpus-per-node=1 \
			--cpus-per-task=24 \
			-o "$SLURM_LOG_DIR/${JOB_NAME}-%j.out" \
			-e "$SLURM_LOG_DIR/${JOB_NAME}-%j.err" \
			--export=ALL \
			${TRAINING_EXTRA_SBATCH:-} \
			"$ROOT_DIR/slurm_run_container.sh" train "$MODEL" "$@" |
			awk '{print $4}'
	)
	echo "Submitted training job: $JOB_ID  (model=$MODEL, GPU x1)"
elif [[ "${1:-}" == "benchmark" ]]; then
	SUBMIT_KIND="benchmark"
	shift
	KIND="${1:?benchmark requires smoke or full}"
	shift || true
	[[ "$KIND" == "smoke" || "$KIND" == "full" ]] || {
		echo "benchmark kind must be smoke or full, got: $KIND" >&2
		exit 1
	}
	NAME="tracin-bench-${KIND}"
	if [[ "${1:-}" != "" && "${1:-}" != -* ]]; then
		NAME="$1"
		shift || true
	fi
	JOB_ID=$(_submit_bench_gpu "$KIND" "" "$NAME")
	WT="$BENCH_TIME_SMOKE"
	[[ "$KIND" == "full" ]] && WT="$BENCH_TIME_FULL"
	echo "Submitted benchmark job: $JOB_ID  (kind=$KIND, GPU x1, walltime=$WT)"
elif [[ "${1:-}" == "benchmark-chain" ]]; then
	SUBMIT_KIND="benchmark-chain"
	SMOKE_NAME="${BENCH_SMOKE_JOB_NAME:-tracin-bench-smoke}"
	FULL_NAME="${BENCH_FULL_JOB_NAME:-tracin-bench-full}"
	SMOKE_ID=$(_submit_bench_gpu smoke "" "$SMOKE_NAME")
	FULL_ID=$(_submit_bench_gpu full "$SMOKE_ID" "$FULL_NAME")
	echo "Submitted benchmark chain:"
	echo "  Smoke job: $SMOKE_ID  ($SMOKE_NAME)"
	echo "  Full job:  $FULL_ID  ($FULL_NAME, runs after smoke exits 0)"
	echo "  (Each job writes to its own \$SCRATCH/tracin_ghost_tool_results/<job_id>/outputs/benchmarks/)"
	echo "Sync both: bash $ROOT_DIR/sync_from_scratch.sh $SMOKE_ID && bash $ROOT_DIR/sync_from_scratch.sh $FULL_ID"
	JOB_ID="$FULL_ID"
else
	CONFIG_PATH="${1:-config.yaml}"
	JOB_NAME="${2:-tracin-ghost}"
	MODE="${3:-index}"
	INPUT_PATH="${4:-${INPUT_PATH:-}}"
	# shellcheck disable=SC2086
	JOB_ID=$(
		sbatch \
			--job-name="$JOB_NAME" \
			-o "$SLURM_LOG_DIR/${JOB_NAME}-%j.out" \
			-e "$SLURM_LOG_DIR/${JOB_NAME}-%j.err" \
			--export=ALL,CONFIG_PATH_INNER="$CONFIG_PATH",MODE="$MODE",INPUT_PATH="$INPUT_PATH" \
			${TRAINING_EXTRA_SBATCH:-} \
			"$ROOT_DIR/slurm_run_container.sh" |
			awk '{print $4}'
	)
	echo "Submitted job: $JOB_ID"
fi

if [[ "$SUBMIT_KIND" == "benchmark-chain" ]]; then
	echo "Track full job: squeue -j $JOB_ID"
else
	echo "Track: squeue -j $JOB_ID"
fi
echo "Logs:  $SLURM_LOG_DIR/*-${JOB_ID}.out (and .err)"
if [[ "$SUBMIT_KIND" == "benchmark" || "$SUBMIT_KIND" == "benchmark-chain" ]]; then
	echo "Benchmark outputs: sync_from_scratch.sh <job_id>  →  outputs/runs/<job_id>/benchmarks/"
fi
