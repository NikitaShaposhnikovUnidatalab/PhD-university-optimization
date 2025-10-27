FROM python:3.11-slim

# Встановлюємо системні залежності для компіляції пакетів
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

# Встановлюємо Python пакети з очищенням тимчасових файлів
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /tmp/* /root/.cache/pip/* && \
    find /usr/local -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true

# Copy application files
COPY . .

EXPOSE 8501 8502

CMD ["streamlit", "run", "app/full/main.py", "--server.port=8501", "--server.address=0.0.0.0"]