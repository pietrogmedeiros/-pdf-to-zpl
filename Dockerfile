FROM python:3.11-slim

# Instala dependências do sistema para PyMuPDF
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    mupdf-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY converter_service.py .

EXPOSE 8000

CMD ["uvicorn", "converter_service:app", "--host", "0.0.0.0", "--port", "8000"]
