#!/usr/bin/env bash
# Start llama.cpp llama-server for the Mistral GGUF at the project root.
# Usage: from repo root,  chmod +x scripts/run_llama_server.sh && ./scripts/run_llama_server.sh
# Optional: LLAMA_SERVER=/path/to/llama-server  LLAMA_EXTRA="-ngl 99"

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GGUF="${ROOT}/mistral-7b-instruct-v0.3-q4_k_m.gguf"
EXE="${LLAMA_SERVER:-llama-server}"

if [[ ! -f "${GGUF}" ]]; then
  echo "GGUF not found: ${GGUF}" >&2
  exit 1
fi

echo "Starting: ${EXE} -m ${GGUF} --alias mistral-7b-local --host 127.0.0.1 --port 8080 -c 8192 ${LLAMA_EXTRA:-}"
exec ${EXE} -m "${GGUF}" --alias mistral-7b-local --host 127.0.0.1 --port 8080 -c 8192 ${LLAMA_EXTRA:-}
