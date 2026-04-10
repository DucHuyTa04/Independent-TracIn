#!/usr/bin/env bash
# Run TracIn Ghost or training: native (Trillium / Compute Canada) or Apptainer.
#
# Default backend: Apptainer if CONTAINER_IMAGE exists, else native modules + conda site-packages.
#
# Usage:
#   ./run_container.sh [config.yaml] [mode]              -> main.py (index|query|full)
#   ./run_container.sh train <model> [train.py args...]  -> testModels/<model>/train.py
#   ./run_container.sh exec -- <cmd> [args...]          -> arbitrary cmd (cwd = repo root)
#   ./run_container.sh native  ...                    -> force native (no Apptainer)
#   ./run_container.sh apptainer ...                  -> force Apptainer (requires .sif)
#
# Native: StdEnv/2023 + TRACIN_EXTRA_MODULES (default cudacore/12.6.3); override if your site uses another CUDA module.
# Environment (optional):
#   CONTAINER_IMAGE     Apptainer .sif (default: ~/containers/python311.sif)
#   SITE_DIR            Host site for Apptainer PYTHONPATH
#   TRAIN_PYTHON        Native interpreter (default: CVMFS gentoo python3.11)
#   TRAINING_ENV_ROOT   Conda env root (default: $SCRATCH/miniconda3/envs/training_env)
#   USE_NATIVE_PYTHON=1 Force native even if .sif exists
#   MAIN_PY_EXTRA       Extra args for main.py legacy path
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$HOME/containers/python311.sif}"
SITE_DIR="${SITE_DIR:-$HOME/containers/py311_site}"
SCRATCH="${SCRATCH:-/scratch/${USER:-${LOGNAME:-user}}}"
TRAINING_ENV_ROOT="${TRAINING_ENV_ROOT:-$SCRATCH/training_influence/Training-Influence-on-Resnet50/venv}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$SCRATCH/training_influence/Training-Influence-on-Resnet50/venv/bin/python}"

_force_native=0
_force_apptainer=0
case "${1:-}" in
native)
	shift
	_force_native=1
	;;
apptainer)
	shift
	_force_apptainer=1
	;;
esac

_use_apptainer() {
	if [[ "$_force_native" -eq 1 ]]; then
		return 1
	fi
	if [[ "$_force_apptainer" -eq 1 ]]; then
		[[ -f "$CONTAINER_IMAGE" ]] || {
			echo "ERROR: CONTAINER_IMAGE not found: $CONTAINER_IMAGE" >&2
			exit 1
		}
		return 0
	fi
	if [[ "${USE_NATIVE_PYTHON:-0}" == "1" ]]; then
		return 1
	fi
	[[ -f "$CONTAINER_IMAGE" ]]
}

_run_apptainer() {
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
			--bind /cvmfs:/cvmfs \
			--bind "$SITE_DIR":/opt/site \
			--bind "$ROOT_DIR":"$ROOT_DIR" \
			"$CONTAINER_IMAGE" \
			"$@"
}

_run_native() {
	module load StdEnv/2023 || true
	local extra="${TRACIN_EXTRA_MODULES:-none}"
	if [[ "$extra" != "none" && "$extra" != "NONE" ]]; then
		# shellcheck disable=SC2086
		module load $extra || {
			echo "WARNING: module load failed: $extra (continuing anyway)" >&2
		}
	fi
	local site="$TRAINING_ENV_ROOT/lib/python3.11/site-packages"
	if [[ ! -d "$site" ]]; then
		echo "WARN: TRAINING_ENV_ROOT site-packages missing: $site" >&2
		echo "      Set TRAINING_ENV_ROOT to your conda env (e.g. .../envs/training_env)." >&2
	fi
	export PYTHONPATH="${site}${PYTHONPATH:+:$PYTHONPATH}"
	# Apptainer uses "python ..."; native already invokes TRAIN_PYTHON — drop redundant "python".
	if [[ "${1:-}" == "python" || "${1:-}" == "python3" ]]; then
		shift
	fi
	(cd "$ROOT_DIR" && exec "$TRAIN_PYTHON" "$@")
}

_run() {
	if _use_apptainer; then
		_run_apptainer "$@"
	else
		_run_native "$@"
	fi
}

# After optional native|apptainer, dispatch (exec/run must come before default config/mode).
case "${1:-}" in
exec | run)
	shift
	_run "$@"
	;;
train)
	shift
	_model="${1:?Usage: $0 [native|apptainer] train <mnist|cifar10_cnn|synth_regression> [train.py args...]}"
	shift
	_run "$ROOT_DIR/testModels/${_model}/train.py" "$@"
	;;
*)
	CONFIG_PATH="${1:-config.yaml}"
	MODE="${2:-index}"
	# shellcheck disable=SC2086
	_run "$ROOT_DIR/main.py" --config "$ROOT_DIR/$CONFIG_PATH" --mode "$MODE" ${MAIN_PY_EXTRA:-}
	;;
esac
