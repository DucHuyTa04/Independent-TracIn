#!/usr/bin/env bash
#SBATCH --job-name=tracin-ghost
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute
## Training jobs: submit_slurm.sh adds --gpus-per-node=1 and --cpus-per-task=24 (Trillium GPU quarter-node).
## For manual GPU training: sbatch --gpus-per-node=1 --cpus-per-task=24 ... slurm_run_container.sh train ...

# Run main.py (index/query/full) or testModels training on Slurm.
# Default: native Python (StdEnv/2023 + TRACIN_EXTRA_MODULES, default cudacore/12.6.3) + conda site-packages.
# Fallback: Apptainer if CONTAINER_IMAGE exists and USE_NATIVE_PYTHON is not 1.
#
# Main:
#   sbatch slurm_run_container.sh [config.yaml]
#   Env: CONFIG_PATH_INNER, MODE, INPUT_PATH, MAIN_MODEL
#
# Training:
#   sbatch slurm_run_container.sh train <model> [train.py args...]
#   Env: TRAIN_DATA_ROOT, TRAIN_OUTPUT_DIR
#
# Benchmarks (GPU):
#   sbatch ... slurm_run_container.sh benchmark <smoke|full>
#   Env: TRAIN_DATA_ROOT, BENCHMARK_OUTPUT_DIR (optional)
#
set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SCRATCH="${SCRATCH:-/scratch/${USER:-${LOGNAME:-user}}}"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-$HOME/containers/python311.sif}"
SITE_DIR="${SITE_DIR:-$HOME/containers/py311_site}"
TRAINING_ENV_ROOT="${TRAINING_ENV_ROOT:-$SCRATCH/training_influence/Training-Influence-on-Resnet50/venv}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$SCRATCH/training_influence/Training-Influence-on-Resnet50/venv/bin/python}"

RUN_DIR="${SCRATCH}/tracin_ghost_tool_jobs/${SLURM_JOB_ID}"
RESULT_DIR="${SCRATCH}/tracin_ghost_tool_results/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR" "$RESULT_DIR/logs" "$RESULT_DIR/outputs/checkpoints" "$RESULT_DIR/data"

# StdEnv is required; CUDA module name/version differs by site (Trillium vs other Alliance nodes).
# If you see "unknown module cudacore/12.6.3" in the job .err, run on a GPU/login node:
#   module spider cuda
# then before sbatch:  export TRACIN_EXTRA_MODULES='cuda/12.x'   (space-separated list is ok)
# or skip extra CUDA modules if your torch wheel bundles libs:  export TRACIN_EXTRA_MODULES=none
_load_native_env() {
	module load StdEnv/2023 || true
	# If TRAIN_PYTHON is a venv python (bundles CUDA), no extra modules needed.
	local extra="${TRACIN_EXTRA_MODULES:-none}"
	if [[ "$extra" != "none" && "$extra" != "NONE" ]]; then
		# shellcheck disable=SC2086
		module load $extra || {
			echo "WARNING: module load failed: $extra (continuing anyway)" >&2
		}
	fi
}

_use_apptainer() {
	if [[ "${USE_NATIVE_PYTHON:-0}" == "1" ]]; then
		return 1
	fi
	[[ -f "$CONTAINER_IMAGE" ]]
}

# Apptainer path (optional)
DEFAULT_BINDS=(
	--bind /cvmfs:/cvmfs
	--bind "$SITE_DIR":/opt/site
	--bind "$ROOT_DIR":"$ROOT_DIR"
	--bind "$RUN_DIR":"$RUN_DIR"
	--bind "$RESULT_DIR":"$RESULT_DIR"
)

_exec_in_container() {
	local ap_flags=()
	while [[ $# -gt 0 ]]; do
		if [[ "$1" == "--" ]]; then
			shift
			break
		fi
		ap_flags+=("$1")
		shift
	done
	module load apptainer/1.3.5
	env \
		-u VIRTUAL_ENV \
		-u PIP_CONFIG_FILE \
		-u SSL_CERT_FILE \
		-u SSL_CERT_DIR \
		-u CURL_CA_BUNDLE \
		PYTHONPATH="$SITE_DIR" \
		apptainer exec \
			--pwd "$ROOT_DIR" \
			"${ap_flags[@]}" \
			"$CONTAINER_IMAGE" \
			"$@"
}

if [[ "${1:-}" == "train" ]]; then
	shift
	TRAIN_MODEL="${1:?Usage: $0 train <mnist|cifar10_cnn|synth_regression> [train.py args...]}"
	shift
	TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-$RESULT_DIR/data}"
	TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$RESULT_DIR/outputs/checkpoints}"
	mkdir -p "$TRAIN_DATA_ROOT" "$TRAIN_OUTPUT_DIR"

	TRAIN_BINDS=(
		--bind /cvmfs:/cvmfs
		--bind "$SITE_DIR":/opt/site
		--bind "$ROOT_DIR":"$ROOT_DIR"
		--bind "$RUN_DIR":"$RUN_DIR"
		--bind "$RESULT_DIR":"$RESULT_DIR"
	)
	_under_result() {
		local p="$1"
		local r="$2"
		[[ "$p" == "$r" || "$p" == "$r/"* ]]
	}
	if ! _under_result "$TRAIN_DATA_ROOT" "$RESULT_DIR"; then
		TRAIN_BINDS+=(--bind "$TRAIN_DATA_ROOT":"$TRAIN_DATA_ROOT")
	fi
	if ! _under_result "$TRAIN_OUTPUT_DIR" "$RESULT_DIR"; then
		TRAIN_BINDS+=(--bind "$TRAIN_OUTPUT_DIR":"$TRAIN_OUTPUT_DIR")
	fi

	echo "Training model=$TRAIN_MODEL -> checkpoints: $TRAIN_OUTPUT_DIR data: $TRAIN_DATA_ROOT"
	if _use_apptainer; then
		_exec_in_container "${TRAIN_BINDS[@]}" -- \
			python "$ROOT_DIR/testModels/${TRAIN_MODEL}/train.py" \
			--output "$TRAIN_OUTPUT_DIR" \
			--data-root "$TRAIN_DATA_ROOT" \
			"$@"
	else
		_load_native_env
		cd "$ROOT_DIR"
		"$TRAIN_PYTHON" "$ROOT_DIR/testModels/${TRAIN_MODEL}/train.py" \
			--output "$TRAIN_OUTPUT_DIR" \
			--data-root "$TRAIN_DATA_ROOT" \
			"$@"
	fi

	echo "Saved job artifacts to: $RESULT_DIR"
	echo "Sync back: bash $ROOT_DIR/sync_from_scratch.sh $SLURM_JOB_ID"
	exit 0
fi

# Benchmarks (GPU): run_all.py with outputs under $RESULT_DIR/outputs/benchmarks
# Usage: sbatch ... slurm_run_container.sh benchmark <smoke|full>
# Env: TRAIN_DATA_ROOT, BENCHMARK_OUTPUT_DIR (optional overrides)
if [[ "${1:-}" == "benchmark" ]]; then
	shift
	BENCH_MODE="${1:?Usage: $0 benchmark <smoke|full>}"
	shift || true
	if [[ "$BENCH_MODE" != "smoke" && "$BENCH_MODE" != "full" ]]; then
		echo "benchmark mode must be smoke or full, got: $BENCH_MODE" >&2
		exit 1
	fi
	TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-$RESULT_DIR/data}"
	BENCH_OUT="${BENCHMARK_OUTPUT_DIR:-$RESULT_DIR/outputs/benchmarks}"
	mkdir -p "$BENCH_OUT" "$TRAIN_DATA_ROOT"

	BENCH_BINDS=(
		--bind /cvmfs:/cvmfs
		--bind "$SITE_DIR":/opt/site
		--bind "$ROOT_DIR":"$ROOT_DIR"
		--bind "$RUN_DIR":"$RUN_DIR"
		--bind "$RESULT_DIR":"$RESULT_DIR"
	)
	_under_result_b() {
		local p="$1"
		local r="$2"
		[[ "$p" == "$r" || "$p" == "$r/"* ]]
	}
	if ! _under_result_b "$TRAIN_DATA_ROOT" "$RESULT_DIR"; then
		BENCH_BINDS+=(--bind "$TRAIN_DATA_ROOT":"$TRAIN_DATA_ROOT")
	fi

	BENCH_EXTRA=()
	if [[ "$BENCH_MODE" == "smoke" ]]; then
		BENCH_EXTRA+=(--scale smoke)
	elif [[ "$BENCH_MODE" == "full" ]]; then
		BENCH_EXTRA+=(--scale full)
	fi

	echo "Benchmark mode=$BENCH_MODE -> $BENCH_OUT (data: $TRAIN_DATA_ROOT)"
	if _use_apptainer; then
		_exec_in_container "${BENCH_BINDS[@]}" -- \
			python "$ROOT_DIR/benchmarks/run_all.py" \
				--output-dir "$BENCH_OUT" \
				--data-root "$TRAIN_DATA_ROOT" \
				--device cuda \
				"${BENCH_EXTRA[@]}"
	else
		_load_native_env
		cd "$ROOT_DIR"
		"$TRAIN_PYTHON" "$ROOT_DIR/benchmarks/run_all.py" \
			--output-dir "$BENCH_OUT" \
			--data-root "$TRAIN_DATA_ROOT" \
			--device cuda \
			"${BENCH_EXTRA[@]}"
	fi

	echo "Saved job artifacts to: $RESULT_DIR"
	echo "Sync back: bash $ROOT_DIR/sync_from_scratch.sh $SLURM_JOB_ID"
	exit 0
fi

CONFIG_PATH="${CONFIG_PATH_INNER:-${1:-config.yaml}}"
TMP_CONFIG="$RUN_DIR/config_job.yaml"
_load_native_env
"$TRAIN_PYTHON" - "$ROOT_DIR/$CONFIG_PATH" "$TMP_CONFIG" "$RESULT_DIR/logs" "$RESULT_DIR/outputs" "$RESULT_DIR" <<'PY'
import sys
import yaml

src, dst, logs_dir, outputs_dir, result_dir = sys.argv[1:6]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

paths = cfg.setdefault("paths", {})
paths["logs_dir"] = logs_dir
paths["outputs_dir"] = outputs_dir
paths["data_root"] = result_dir + "/data"

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

MAIN_EXTRA=()
[[ -n "${INPUT_PATH:-}" ]] && MAIN_EXTRA+=(--input "$INPUT_PATH")
[[ -n "${MAIN_MODEL:-}" ]] && MAIN_EXTRA+=(--model "$MAIN_MODEL")

if _use_apptainer; then
	_exec_in_container "${DEFAULT_BINDS[@]}" -- \
		python "$ROOT_DIR/main.py" --config "$TMP_CONFIG" --mode "${MODE:-index}" "${MAIN_EXTRA[@]}"
else
	cd "$ROOT_DIR"
	"$TRAIN_PYTHON" "$ROOT_DIR/main.py" --config "$TMP_CONFIG" --mode "${MODE:-index}" "${MAIN_EXTRA[@]}"
fi

echo "Saved job artifacts to: $RESULT_DIR"
echo "Sync back to workspace with: bash $ROOT_DIR/sync_from_scratch.sh $SLURM_JOB_ID"
