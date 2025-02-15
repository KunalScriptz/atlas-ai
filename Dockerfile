FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Rust-based Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml ./
RUN uv pip install --system --no-cache -e .

COPY . .

EXPOSE 9734 8501

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "9734"]
