FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install .
CMD ["uvicorn", "{{package_name}}.api:app", "--host", "0.0.0.0", "--port", "8080"]
