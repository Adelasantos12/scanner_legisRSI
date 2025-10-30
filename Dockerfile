# Usa una imagen base de Python slim para un tamaño menor
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia solo el archivo de dependencias primero para aprovechar el cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el resto de tu código al contenedor
COPY . .

# --- Pasos de Build ---
# 1. Genera los archivos de datos necesarios
RUN python scripts/generate_metadata_catalog.py
RUN python scripts/generate_training_data.py

# 2. Entrena el modelo. Esto creará la carpeta `models/` que la app necesita.
RUN python scripts/train_classifier_latam.py

# Expone el puerto que usará Gunicorn (Render lo necesita)
EXPOSE 10000

# El comando que se ejecutará para iniciar la aplicación en producción
# - Workers/Threads: Configuración para manejar concurrencia.
# - Timeout: Extendido a 3 minutos (180s) para dar tiempo a que el modelo BERT cargue en memoria.
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "4", "--timeout", "180", "app:app"]
