FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Verify tasks.json exists
RUN ls -la /app/tasks.json && echo "✓ tasks.json found in container"

EXPOSE 8000

# Pre-warm the environment on startup for faster responses
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run with verbose logging and unbuffered output
# --timeout-keep-alive controls how long uvicorn waits before closing idle connections
# --timeout-notify controls shutdown grace period
CMD ["python", "-u", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info", "--timeout-keep-alive", "5"]
