# Use slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system deps (important for pandas, psycopg2, etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app/ app/
COPY models/ models/

# Expose FastAPI port
EXPOSE 8000

# Run API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]