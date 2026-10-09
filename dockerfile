FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY scripts ./scripts
COPY data/synthetic ./data/synthetic
RUN python scripts/train.py
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
