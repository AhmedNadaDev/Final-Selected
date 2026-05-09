FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps (ffmpeg required for video + audio muxing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 curl git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CLIP (optional — fall back gracefully if install fails)
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git || \
    echo "CLIP install failed — reranker / CLIP-SIM metric will be disabled"

COPY . .

RUN mkdir -p outputs

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
