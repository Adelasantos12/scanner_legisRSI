# Usa una imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de dependencias e instálalas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el resto de tu código al contenedor
COPY . .

# --- Pasos de Build ---
# Asegúrate de que los datos necesarios estén generados
RUN python scripts/generate_metadata_catalog.py
RUN python scripts/generate_training_data.py
# NOTA: El modelo ya está entrenado, así que asegúrate de que la carpeta 'models'
# se copie al hacer el deploy. Si es muy grande para el repo,
# necesitarías descargarla de un almacenamiento en la nube aquí.

# Expone el puerto que usará Gunicorn
EXPOSE 10000

# El comando que se ejecutará para iniciar la aplicación
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"
