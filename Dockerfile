FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git git-lfs ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
    "huggingface_hub<0.25" \
    -r requirements.txt \
    gradio==4.44.0 \
    "uvicorn>=0.14.0" \
    "websockets>=10.4" \
    spaces

COPY . .

RUN mkdir -p /home/user && ln -s /app /home/user/app

CMD ["python", "app.py"]
