FROM python:3.12-slim

WORKDIR /app

# Install system deps if needed (optional, but useful for psycopg2 etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project code into the image
COPY . .

# Default command can be overridden by docker-compose
CMD ["bash"]