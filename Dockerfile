# FROM python:3.11-slim
# WORKDIR /app
# RUN apt-get update && apt-get install -y \
#     ffmpeg \
#     && rm -rf /var/lib/apt/lists/*
# COPY requirements.txt .
# RUN pip install -r requirements.txt
# COPY . .
# CMD ["python", "app.py"]
FROM ultralytics/ultralytics:latest-arm64
WORKDIR /app
# Verify ffmpeg is present; install only if missing
RUN which ffmpeg || (apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]
