#!/bin/bash
set -e

# Run migrations if alembic is present
if [ -f alembic.ini ] || [ -f alembic.ini.example ]; then
  if [ -f alembic.ini ]; then
    uv run alembic upgrade head
  else
    cp alembic.ini.example alembic.ini
    sed -i "s|^sqlalchemy.url =.*|sqlalchemy.url = $DATABASE_URL|" alembic.ini
    uv run alembic upgrade head
  fi
fi

# Start the bot
uv run main.py
