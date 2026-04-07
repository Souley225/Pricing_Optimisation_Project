# Dockerfile for Hugging Face Spaces (API + Streamlit combined)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, configs, and models
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV API_URL=http://localhost:8000

# Expose Streamlit port (HF Spaces default)
EXPOSE 7860

# Copy and set entrypoint
COPY start.sh ./
RUN chmod +x start.sh

CMD ["./start.sh"]
