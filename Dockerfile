# Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy code ONLY (exclude .env, .gitignore, etc.)
COPY app/ /app/

# Expose ports
EXPOSE 9000 9001 8000

ENV PYTHONUNBUFFERED=1

# Local dev default
#CMD ["python", "app/launcher.py", "--tasks", "5"]

# Run launcher (starts both agents)
CMD ["python", "app/launcher.py"]