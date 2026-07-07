FROM python:3.13-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir uv && uv sync --group dev
CMD ["uv", "run", "pytest"]
