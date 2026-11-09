FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy package specifications
COPY pyproject.toml README.md ./

# Install production dependencies for the API
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn pydantic scikit-learn joblib numpy

# Copy application source code and models
COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
