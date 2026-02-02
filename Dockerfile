# Decision Engine Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (git for pip git+ installs, libpq for psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY decision_engine/ /app/decision_engine/
COPY config/ /app/config/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the service
CMD ["python", "-m", "decision_engine.main"]
