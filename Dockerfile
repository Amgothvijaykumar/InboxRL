FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Verify tasks.json exists
RUN ls -la /app/tasks.json && echo "✓ tasks.json found in container"

EXPOSE 8000

# Run with verbose logging and unbuffered output
CMD ["python", "-u", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
