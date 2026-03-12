# Use Python 3.11 slim for smaller image (InsightFace works on CPU)
FROM python:3.11-slim

# Install system deps: runtime libs for OpenCV/InsightFace + build tools for insightface (Cython/C++)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY model_creat.py .

# Railway sets PORT; default for local Docker
ENV PORT=8001
EXPOSE 8001

# Run with PORT from environment so Railway can inject it (shell form for env expansion)
CMD ["sh", "-c", "uvicorn model_creat:app --host 0.0.0.0 --port ${PORT}"]
