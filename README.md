# Clasificador de Leyes del Reglamento Sanitario Internacional (RSI)

Este proyecto es una aplicación web multi-página construida con Streamlit que proporciona herramientas para el análisis de marcos jurídicos en relación con el Reglamento Sanitario Internacional (RSI).

## Módulos

1.  **📋 Clasificador de Marco Jurídico Nacional:**
    *   Permite subir una lista masiva de leyes de un país.
    *   Utiliza un modelo de `sentence-transformers` para clasificar cada ley por su sector RSI más probable y asignar un score de relevancia.
    *   Proporciona visualizaciones y un resumen estadístico del marco jurídico.
    *   Permite seleccionar las leyes más relevantes para un análisis más profundo.

2.  **⚖️ Módulo de Escaneo Profundo de Leyes:**
    *   Recibe las leyes seleccionadas o texto nuevo.
    *   Utiliza un modelo BERT para un análisis semántico detallado, identificando la función RSI dominante.
    *   Extrae palabras clave y metadatos de la ley.

---

## Configuración y Ejecución Local

Siga estos pasos para configurar el entorno, generar los artefactos y ejecutar la aplicación.

### 1. Prerrequisitos

-   Python 3.9 o superior
-   `pip` (gestor de paquetes de Python)

### 2. Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### 3. Build de Artefactos (¡Paso Crucial!)

La aplicación necesita archivos de datos, centroides y un modelo entrenado para funcionar. Ejecute los siguientes comandos en orden:
```bash
# 1. Generar archivos de datos
python scripts/generate_metadata_catalog.py
python scripts/generate_training_data.py

# 2. Generar centroides para el clasificador
python scripts/generate_centroids.py

# 3. Entrenar el modelo de análisis profundo (esto crea la carpeta `models/`)
python scripts/train_classifier_latam.py
```
**Nota:** El paso de entrenamiento puede consumir mucha memoria RAM (>4GB). Si falla en su entorno local, es una limitación de recursos. El deploy en la nube (ver abajo) solucionará esto.

### 4. Ejecutar la Aplicación Streamlit

Una vez que los artefactos estén construidos, inicie la aplicación:
```bash
streamlit run scanner_app.py
```
La aplicación se abrirá automáticamente en su navegador.

---

## Deploy en Render o Railway (Recomendado)

Este proyecto incluye un `Dockerfile` para un despliegue sencillo en plataformas PaaS.

### 1. Configuración del Servicio

-   Cree un nuevo "Web Service" en su plataforma (Render, Railway, etc.).
-   Conéctelo a su repositorio de GitHub.
-   **Runtime**: Elija `Docker`. La plataforma detectará el `Dockerfile`.
-   **Plan de Instancia**: **Importante:** Debido a los modelos de ML, el plan gratuito no será suficiente. Seleccione un plan con **al menos 4GB de RAM** para asegurar que el build (que incluye el entrenamiento del modelo) y la ejecución de la aplicación no fallen.

### 2. Desplegar

-   Inicie el deploy. La plataforma construirá la imagen de Docker, ejecutando todos los scripts de build. Este primer build puede tardar varios minutos.
-   Una vez finalizado, su aplicación estará en vivo en una URL pública.
