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

# Synchronize dependencies (FinanceDataReader required for 15:20 candle sync / 09:00 opens)
RUN uv sync --frozen && uv run python -c "import FinanceDataReader"

# Copy source code, scripts, and research models
COPY src ./src
COPY research ./research
COPY scripts ./scripts
COPY docs ./docs

# Set PYTHONPATH (compose may override to include /app for scripts)
ENV PYTHONPATH=/app/src:/app

# Default entrypoint command: daemon scheduler
CMD ["uv", "run", "python", "-m", "kr_stock.scheduler"]
