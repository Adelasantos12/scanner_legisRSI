import yaml

# Definición de patrones de regex para LATAM
patterns = {
    "vigilancia": [
        "vigilancia epidemiológica",
        "sistema nacional de vigilancia",
        "notificación obligatoria"
    ],
    "deteccion": [
        "diagnóstico",
        "laboratorio",
        "análisis de muestras"
    ],
    # Añadir más funciones y patrones aquí
}

# Guardar en un archivo YAML en el directorio de datos
output_path = "data/regex_patterns_latam.yaml"
with open(output_path, "w", encoding="utf-8") as f:
    yaml.dump(patterns, f, allow_unicode=True, indent=2)

print(f"✅ Patrones de regex para LATAM guardados en: {output_path}")
