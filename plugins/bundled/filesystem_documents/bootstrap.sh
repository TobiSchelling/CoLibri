#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PY="${VENV_DIR}/bin/python"

if command -v python3.12 >/dev/null 2>&1; then
  python3.12 -m venv "${VENV_DIR}"
else
  python3 -m venv "${VENV_DIR}"
fi
set +e
"${PY}" -m pip install -r "${ROOT_DIR}/requirements.txt"
status=$?
set -e

if [[ $status -ne 0 ]]; then
  echo "filesystem_documents: dependency install failed." >&2
  echo " - If you are offline / sandboxed, download wheels elsewhere and retry with:" >&2
  echo "   ${PY} -m pip install --no-index --find-links /path/to/wheels -r ${ROOT_DIR}/requirements.txt" >&2
  exit $status
fi

echo "filesystem_documents: venv ready at ${VENV_DIR}" >&2
