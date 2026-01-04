#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

setup_venv() {
  local target_dir="$1"
  local requirements_file="$2"
  local abs_dir="${ROOT_DIR}/${target_dir}"

  if [[ ! -d "${abs_dir}" ]]; then
    echo "[setup_internals] Skipping missing directory: ${target_dir}"
    return 0
  fi
  if [[ ! -f "${abs_dir}/${requirements_file}" ]]; then
    echo "[setup_internals] Skipping ${target_dir}: requirements file '${requirements_file}' not found."
    return 0
  fi

  echo "[setup_internals] Creating venv for ${target_dir} with uv"
  uv venv "${abs_dir}/.venv"
  uv pip install --python "${abs_dir}/.venv/bin/python" --no-cache -r "${abs_dir}/${requirements_file}"
}

setup_venv "audio_generation/piper_side/internals" "requirements.txt"
setup_venv "audio_generation/suno_side/internals" "requirements.txt"
setup_venv "audio_generation/kokoro_side/internals" "requirements.txt"
setup_venv "audio_generation/elevenlabs_side/internals" "requirements.txt"
setup_venv "data_generation/keyphrase_finding/internal" "requirements.txt"

