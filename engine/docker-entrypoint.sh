#!/bin/sh
set -eu

alembic -c alembic.ini.example upgrade head
python -m scripts.preload_objects

exec uvicorn paradox_engine.app:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}"
