FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p data logo

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8501

# Expose the port
EXPOSE 8501

# Health check endpoint
COPY <<EOF /app/health.py
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()

def start_health_server():
    server = HTTPServer(('0.0.0.0', 8081), HealthHandler)
    server.serve_forever()

if __name__ == '__main__':
    threading.Thread(target=start_health_server, daemon=True).start()
    time.sleep(999999)
EOF

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]

