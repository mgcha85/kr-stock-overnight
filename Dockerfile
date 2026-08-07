FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt-get/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Synchronize dependencies
RUN uv sync --frozen

# Copy source code and research models
COPY src ./src
COPY research ./research
COPY docs ./docs

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Default entrypoint command: daemon scheduler
CMD ["uv", "run", "python", "-m", "kr_stock.scheduler"]
