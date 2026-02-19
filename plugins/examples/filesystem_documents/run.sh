#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PY="${VENV_DIR}/bin/python"

if [[ -x "${PY}" ]]; then
  exec "${PY}" "${ROOT_DIR}/plugin.py"
fi

exec python3 "${ROOT_DIR}/plugin.py"
