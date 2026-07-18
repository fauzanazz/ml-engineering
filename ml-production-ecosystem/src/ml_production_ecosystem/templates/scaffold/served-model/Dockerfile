FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO

WORKDIR /app
RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md ./
COPY {{package_name}} ./{{package_name}}

RUN pip install --no-cache-dir . && python -m compileall -q {{package_name}}

USER app
EXPOSE 8000

CMD ["sh", "-c", "uvicorn {{package_name}}.api:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
