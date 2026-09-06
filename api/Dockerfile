FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serve.py .
# Build context is the project root so models/ and data/processed/ can be copied in:
COPY models /app_models
COPY data/processed /app_data
ENV MODELS_DIR=/app_models
ENV DATA_DIR=/app_data
EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
