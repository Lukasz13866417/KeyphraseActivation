#!/usr/bin/env bash
set -euo pipefail

MODE="${APP_MODE:-}"
if [[ -z "${MODE}" && $# -gt 0 ]]; then
  MODE="$1"
  shift
fi
MODE="${MODE:-web}"

PORT="${PORT:-8000}"
RUN_MIGRATIONS="${RUN_MIGRATIONS:-1}"
DJANGO_DB_PATH="${DJANGO_SQLITE_PATH:-}"

mkdir -p /app/samples /app/artifacts /app/db

python - <<'PY'
from db import db_api
try:
    db_api.init_db()
except Exception as exc:
    print(f"[entrypoint] DB init warning: {exc}")
PY

if [[ -n "${DJANGO_DB_PATH}" ]]; then
  mkdir -p "$(dirname "${DJANGO_DB_PATH}")"
fi

case "${MODE}" in
  web)
    if [[ "${RUN_MIGRATIONS}" != "0" ]]; then
      python webui/manage.py migrate --noinput
    fi
    exec python webui/manage.py runserver "0.0.0.0:${PORT}"
    ;;
  cli)
    exec python main.py
    ;;
  bash)
    exec /bin/bash "$@"
    ;;
  *)
    if [[ $# -gt 0 ]]; then
      exec "${MODE}" "$@"
    else
      exec "${MODE}"
    fi
    ;;
esac

