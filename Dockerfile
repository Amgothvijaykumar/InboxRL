FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HuggingFace Spaces requires non-root user with UID 1000
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

ENTRYPOINT ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "5"]
