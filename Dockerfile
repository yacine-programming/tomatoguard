FROM python:3.11-slim

# Installer dépendances système nécessaires (OpenCV pour YOLO)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier les fichiers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces utilise le port 7860
ENV PORT=7860
EXPOSE 7860

# Lancer le serveur
CMD ["gunicorn", "app:app", "--timeout", "300", "--workers", "1", "--bind", "0.0.0.0:7860"]