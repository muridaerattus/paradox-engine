#!/bin/bash
set -e

# Use DATABASE_URL from environment, default to SQLite if not set
export DATABASE_URL="${DATABASE_URL:-sqlite+aiosqlite:///./paradox.db}"

# If using SQLite, ensure paradox.db is not overwritten
if [[ "$DATABASE_URL" == sqlite+aiosqlite* ]]; then
  if [ ! -f ./paradox.db ]; then
    touch ./paradox.db
  fi
fi

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
