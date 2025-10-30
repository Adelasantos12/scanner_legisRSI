# Clasificador de Leyes del Reglamento Sanitario Internacional (RSI)

Este proyecto es un prototipo funcional de una aplicación web diseñada para clasificar fragmentos de leyes mexicanas según las funciones del Reglamento Sanitario Internacional (RSI 2005).

## Características

-   **Análisis de Texto**: Permite pegar texto directamente o subir archivos (`.txt`, `.docx`, `.pdf`).
-   **Clasificación Automática**: Identifica la función RSI principal, el sector probable, la autoridad legal y la ley asociada.
-   **Visualización de Resultados**: Muestra un resumen del análisis, las palabras clave detectadas y un gráfico de radar con la distribución de las funciones RSI.
-   **Exportación**: Permite descargar los resultados del análisis en formato `.csv`.
-   **Modularidad**: Diseñado para ser extensible a otros países en el futuro.

## Instalación

Siga estos pasos para configurar el entorno de desarrollo y ejecutar la aplicación.

### Prerrequisitos

-   Python 3.8 o superior
-   `pip` (gestor de paquetes de Python)

### Pasos

1.  **Clonar el repositorio (si aplica)**:
    ```bash
    git clone <url-del-repositorio>
    cd <nombre-del-repositorio>
    ```

2.  **Crear un entorno virtual (recomendado)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instalar las dependencias**:
    El archivo `requirements.txt` contiene todas las bibliotecas de Python necesarias.
    ```bash
    pip install -r requirements.txt
    ```

## Uso

Una vez que las dependencias estén instaladas, puede iniciar la aplicación.

1.  **Ejecutar el servidor Flask**:
    ```bash
    python app.py
    ```

2.  **Acceder a la aplicación**:
    Abra su navegador web y vaya a la siguiente URL:
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

    La aplicación estará lista para recibir texto o archivos para su análisis.
