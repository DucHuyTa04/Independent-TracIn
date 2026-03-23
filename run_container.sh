#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$HOME/containers/python311.sif}"
SITE_DIR="${SITE_DIR:-$HOME/containers/py311_site}"
CONFIG_PATH="${1:-config.yaml}"
MODE="${2:-index}"

module load apptainer/1.3.5

env \
  -u VIRTUAL_ENV \
  -u PIP_CONFIG_FILE \
  -u SSL_CERT_FILE \
  -u SSL_CERT_DIR \
  -u CURL_CA_BUNDLE \
  PYTHONPATH="$SITE_DIR" \
  apptainer exec \
    --bind /cvmfs:/cvmfs \
    --bind "$SITE_DIR":/opt/site \
    --bind "$ROOT_DIR":"$ROOT_DIR" \
    "$CONTAINER_IMAGE" \
    python "$ROOT_DIR/main.py" --config "$ROOT_DIR/$CONFIG_PATH" --mode "$MODE"
