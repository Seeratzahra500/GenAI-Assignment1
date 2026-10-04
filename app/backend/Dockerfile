FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MODELS_DIR=/models SAMPLES_DIR=/app/samples
EXPOSE 8000
CMD ["./start.sh"]
