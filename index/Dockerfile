FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV HOST=0.0.0.0
ENV PORT=8000
ENV FASTEMBED_CACHE_PATH=/models/fastembed
ENV HF_HOME=/models/huggingface

RUN mkdir -p /models/fastembed /models/huggingface

# Скачиваем sparse модель во время сборки (в контейнере нет интернета)
RUN python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25')"

EXPOSE 8000

# Запускаем через CLI uvicorn — надёжнее чем programmatic API с workers
CMD ["sh", "-c", "uvicorn main:app --host $HOST --port $PORT --workers 4"]
