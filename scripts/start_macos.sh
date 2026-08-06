#!/bin/bash
# Entrypoint for the launchd agent on macOS.
#
# launchd does not run a shell, so it cannot expand variables or resolve the
# venv path. This wrapper does that, then execs uvicorn in place so launchd
# tracks the real server process rather than a parent shell.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "missing venv interpreter at $PYTHON" >&2
    exit 1
fi

mkdir -p logs

# uvicorn takes host/port as CLI flags, but they are configured in .env, which
# only pydantic-settings reads. Ask Settings for the resolved values so .env
# stays the single source of truth. Everything else (device, auth, encryption)
# is read from .env by the app itself.
read -r HOST PORT <<<"$("$PYTHON" -c 'from clip_classifier_service.settings import Settings
s = Settings()
print(s.host, s.port)')"

exec "$PYTHON" -m uvicorn clip_classifier_service.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1
