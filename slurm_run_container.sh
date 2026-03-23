#!/usr/bin/env bash
#SBATCH --job-name=tracin-ghost
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
CONFIG_PATH="${CONFIG_PATH_INNER:-${1:-config.yaml}}"

RUN_DIR="${SCRATCH}/tracin_ghost_tool_jobs/${SLURM_JOB_ID}"
RESULT_DIR="${SCRATCH}/tracin_ghost_tool_results/${SLURM_JOB_ID}"

mkdir -p "$RUN_DIR" "$RESULT_DIR"
mkdir -p "$RESULT_DIR/logs" "$RESULT_DIR/outputs/checkpoints"

TMP_CONFIG="$RUN_DIR/config_job.yaml"
python - "$ROOT_DIR/$CONFIG_PATH" "$TMP_CONFIG" "$RESULT_DIR/logs" "$RESULT_DIR/outputs" "$RESULT_DIR" <<'PY'
import sys
import yaml

src, dst, logs_dir, outputs_dir, result_dir = sys.argv[1:6]
with open(src, "r", encoding="utf-8") as f:
	cfg = yaml.safe_load(f)

cfg["paths"]["logs_dir"] = logs_dir
cfg["paths"]["outputs_dir"] = outputs_dir
cfg["paths"]["data_root"] = result_dir + "/data"

with open(dst, "w", encoding="utf-8") as f:
	yaml.safe_dump(cfg, f, sort_keys=False)
PY

module load apptainer/1.3.5

# main.py supports modes: index, query, full
env \
	-u VIRTUAL_ENV \
	-u PIP_CONFIG_FILE \
	-u SSL_CERT_FILE \
	-u SSL_CERT_DIR \
	-u CURL_CA_BUNDLE \
	PYTHONPATH="$HOME/containers/py311_site" \
	apptainer exec \
		--bind /cvmfs:/cvmfs \
		--bind "$HOME/containers/py311_site":/opt/site \
		--bind "$ROOT_DIR":"$ROOT_DIR" \
		--bind "$RUN_DIR":"$RUN_DIR" \
		--bind "$RESULT_DIR":"$RESULT_DIR" \
		"$HOME/containers/python311.sif" \
		python "$ROOT_DIR/main.py" --config "$TMP_CONFIG" --mode "${MODE:-index}" \
			$([ -n "${INPUT_PATH:-}" ] && echo "--input $INPUT_PATH" || true)

echo "Saved job artifacts to: $RESULT_DIR"
echo "Sync back to workspace with: bash $ROOT_DIR/sync_from_scratch.sh $SLURM_JOB_ID"
