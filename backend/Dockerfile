FROM python:3.10-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/

# Copy model file (or mount as volume in docker-compose)
# COPY garbage_efficientnet_model.h5 .

# Environment
ENV MODEL_PATH=/app/model/garbage_efficientnet_model.h5
ENV LOG_LEVEL=INFO
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
