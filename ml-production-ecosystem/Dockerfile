FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY templates ./templates

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["foundation-serve-recommender", "--host", "0.0.0.0", "--port", "8000"]
