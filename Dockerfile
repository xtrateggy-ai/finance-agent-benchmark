FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME=/app/cache

# System dependencies (needed by many AI libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies first (better caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY utils/ ./utils/
COPY tools/ ./tools/
# Copy dataset into image
COPY data/ ./data/

# Create runtime directories
RUN mkdir -p /app/cache

# Default command (same as your local run)
ENTRYPOINT ["python", "-m", "src.server"]
