# Image Python officielle légère
FROM python:3.10-slim

# Dossier de travail dans le container
WORKDIR /app

# Copier les fichiers requirements
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . .

# Exposer le port FastAPI
EXPOSE 8000

# Lancer l’API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]