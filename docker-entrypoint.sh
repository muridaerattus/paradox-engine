#!/bin/bash
set -e

# Run migrations if alembic is present
if [ -f alembic.ini ] || [ -f alembic.ini.example ]; then
  if [ ! -f alembic.ini ]; then
    cp alembic.ini.example alembic.ini
  fi
  sed -i "s|^sqlalchemy.url =.*|sqlalchemy.url = $DATABASE_URL|" alembic.ini
  uv run alembic upgrade head
fi

touch paradox.db

uv run preload_objects.py

# Start the bot
uv run main.py
