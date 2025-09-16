FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libhdf5-dev \
    git \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install NPEET for entropy calculations
RUN git clone https://github.com/gregversteeg/NPEET.git \
    && pip install --no-cache-dir ./NPEET \
    && rm -rf NPEET

# Create necessary directories
RUN mkdir -p ./instance

# Set environment variables
ENV ENVIRONMENT_FILE=".env"
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY src/ ./src/
COPY src/gunicorn_config.py ./

# Expose port
EXPOSE 1050

# Set default command
ENTRYPOINT ["gunicorn", "--config", "gunicorn_config.py", "src.app:server"]


