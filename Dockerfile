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
RUN python scripts/generate_centroids.py

# 2. Entrena el modelo. Esto creará la carpeta `models/` que la app necesita.
RUN python scripts/train_classifier_latam.py

# El comando que se ejecutará para iniciar la aplicación con Streamlit.
# Usamos la "shell form" para que la variable de entorno $PORT sea interpretada.
CMD streamlit run scanner_app.py --server.port=$PORT --server.address=0.0.0.0
