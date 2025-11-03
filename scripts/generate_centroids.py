import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def generate_centroids():
    """
    Genera embeddings de centroides para cada sector RSI a partir de palabras clave definidas.
    """
    print("🚀 Iniciando la generación de centroides de sector RSI...")

    # --- 1. Cargar recursos ---
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # Usaremos las funciones RSI de la ontología como base para los sectores
    ontology_path = os.path.join('data', 'ontology_rsi_mexico.json')
    with open(ontology_path, 'r', encoding='utf-8') as f:
        ontology = json.load(f)

    # Usaremos las funciones como nuestras "palabras clave" representativas de cada sector
    sector_keywords = {func: [func.replace("_", " ")] for func in ontology['enumerations']['FunctionsRSI']}

    # --- 2. Generar centroides ---
    centroids = {}
    for sector, keywords in sector_keywords.items():
        print(f"  -> Procesando sector: {sector}")
        # Generar embeddings para las palabras clave del sector
        embeddings = model.encode(keywords)
        # Calcular el centroide (embedding promedio)
        centroid = np.mean(embeddings, axis=0)
        centroids[sector] = centroid.tolist() # Convertir a lista para serialización JSON

    # --- 3. Guardar centroides ---
    output_path = os.path.join('data', 'sector_centroids.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(centroids, f, indent=4)

    print(f"✅ Centroides guardados exitosamente en: {output_path}")

if __name__ == "__main__":
    generate_centroids()
