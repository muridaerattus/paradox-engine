FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project


FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    DATABASE_URL=sqlite+aiosqlite:////data/paradox.db \
    CLASS_QUIZ_FILENAME=/app/class_quiz.json \
    ASPECT_QUIZ_FILENAME=/app/aspect_quiz.json \
    PROMPTS_DIRECTORY=/app/prompts

RUN useradd --create-home --uid 10001 app \
    && mkdir /data \
    && chown app:app /data

COPY --from=builder /app/.venv /app/.venv
COPY --chown=app:app . /app

RUN chmod +x /app/docker-entrypoint.sh

USER app

EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=3)"]

ENTRYPOINT ["/app/docker-entrypoint.sh"]
