#!/bin/bash
set -e

# Set up alembic.ini from example if not present
if [ ! -f "/app/alembic.ini" ]; then
    cp /app/alembic.ini.example /app/alembic.ini
    if [ -n "$DATABASE_URL" ]; then
        sed -i "s|^sqlalchemy.url.*$|sqlalchemy.url = $DATABASE_URL|" /app/alembic.ini
    fi
fi

# Activate virtual environment
source /app/.venv/bin/activate

# Run Alembic migrations using uv
uv run alembic upgrade head

# Start the bot using uv
exec uv run main.py
