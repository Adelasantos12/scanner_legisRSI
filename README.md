# Clasificador de Leyes del Reglamento Sanitario Internacional (RSI)

Este proyecto es una aplicación web que utiliza un modelo de Machine Learning (BERT) para clasificar fragmentos de leyes mexicanas según las funciones del Reglamento Sanitario Internacional (RSI 2005).

La aplicación tiene "conciencia jurídica", lo que significa que distingue entre la identidad de la ley analizada y el contenido temático o "insight sectorial dominante".

## Características

-   **Análisis de Texto**: Permite pegar texto o subir archivos (`.txt`, `.docx`, `.pdf`).
-   **Modelo BERT**: Utiliza un modelo BERT multilingüe para un análisis semántico preciso.
-   **Conciencia Jurídica**: Identifica la ley de origen, autoridad y fecha de publicación.
-   **Visualización Completa**: Muestra un resumen del análisis, palabras clave de regex y un gráfico de radar.
-   **Exportación**: Permite descargar los resultados en formato `.csv`.

---

## Configuración y Ejecución Local

Siga estos pasos para configurar el entorno, generar los artefactos necesarios y ejecutar la aplicación en su máquina local.

### 1. Prerrequisitos

-   Python 3.9 o superior
-   `pip` (gestor de paquetes de Python)

### 2. Instalación de Dependencias

Clone el repositorio, cree un entorno virtual (recomendado) y luego instale las dependencias.
```bash
# (Opcional) Crear y activar un entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar bibliotecas
pip install -r requirements.txt
```

### 3. Build de Artefactos (¡Paso Crucial!)

La aplicación necesita archivos de datos y un modelo entrenado para funcionar. Estos no se guardan en Git y deben generarse localmente.

Ejecute los siguientes comandos en orden:
```bash
# 1. Generar el catálogo de metadatos JSON
python scripts/generate_metadata_catalog.py

# 2. Generar el archivo de datos de entrenamiento CSV
python scripts/generate_training_data.py

# 3. Entrenar el modelo (esto creará la carpeta `models/`)
python scripts/train_classifier_latam.py
```

### 4. Ejecutar la Aplicación

Una vez que los artefactos estén construidos, inicie el servidor Flask.
```bash
python app.py
```

Abra su navegador y vaya a [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## Deploy en Render (Recomendado)

Este proyecto incluye un `Dockerfile` para un despliegue sencillo y robusto en plataformas PaaS como Render.

### 1. Crear una cuenta en Render

-   Vaya a [render.com](https://render.com/), cree una cuenta y conéctela a su repositorio de GitHub.

### 2. Crear un Nuevo "Web Service"

-   En el dashboard de Render, haga clic en "New" -> "Web Service".
-   Seleccione su repositorio.

### 3. Configuración del Servicio

-   **Runtime**: Elija `Docker`. Render detectará automáticamente el `Dockerfile`.
-   **Plan de Instancia**: **Importante:** El modelo BERT consume una cantidad significativa de RAM. El plan gratuito no será suficiente. Seleccione un plan de pago con al menos 2GB de RAM (por ejemplo, "Starter" o superior) para asegurar que la aplicación pueda iniciarse.
-   **Health Check Path**: Puede usar la ruta por defecto `/`.

### 4. Desplegar

-   Haga clic en "Create Web Service". Render comenzará el proceso de build, que puede tardar varios minutos ya que incluye la instalación de dependencias y el entrenamiento del modelo.
-   Una vez finalizado, su aplicación estará disponible en una URL pública.
