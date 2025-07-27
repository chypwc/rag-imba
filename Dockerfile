# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for implicit library
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libopenblas-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Expose port
EXPOSE 8000

# Use uvicorn to start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
