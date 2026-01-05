# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the benchmark script
COPY median_benchmark.py .

# Run the benchmark when container starts
CMD ["python", "median_benchmark.py"]
