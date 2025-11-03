import re
import json
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer, util

# --- Carga de Modelos y Datos ---
# Esta sección debe ser llamada desde la app principal de Streamlit con @st.cache_resource

def load_analysis_resources():
    """Carga todos los modelos y datos necesarios para el análisis profundo."""
    from transformers import BertTokenizerFast, BertForSequenceClassification

    # Modelo BERT
    model_dir = "models/bert_rsi_latam"
    bert_tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    bert_model = BertForSequenceClassification.from_pretrained(model_dir)

    # Archivos de datos
    with open('data/ontology_rsi_mexico.json', 'r', encoding='utf-8') as f:
        ontology = json.load(f)
    with open('data/regex_patterns_mexico.yaml', 'r', encoding='utf-8') as f:
        regex_patterns = yaml.safe_load(f)
    with open('data/legal_metadata_catalog.json', 'r', encoding='utf-8') as f:
        legal_catalog = json.load(f)

    return bert_tokenizer, bert_model, ontology, regex_patterns, legal_catalog

# --- Lógica de Análisis ---

def clean_text(text):
    text = text.lower()
    return re.sub(r'\s+', ' ', text)

def legal_metadata_resolver(text, legal_catalog):
    cleaned_text = clean_text(text)
    for key, metadata in legal_catalog.items():
        if re.search(r'\b' + re.escape(metadata['nombre'].lower()) + r'\b', cleaned_text) or \
           re.search(r'\b' + re.escape(metadata['sigla_local'].lower()) + r'\b', cleaned_text):
            return {
                "ley_analizada": f"{metadata['nombre']} ({metadata['autoridad_responsable_principal']['sigla_local']})",
                "autoridad_legal": metadata['autoridad_responsable_principal']['nombre_local'],
                "dof": metadata.get('dof_publicacion', 'No disponible')
            }
    return { "ley_analizada": "No identificada", "autoridad_legal": "No identificada", "dof": "No disponible" }

def semantic_analyzer(text, bert_tokenizer, bert_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)

    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        logits = bert_model(**inputs).logits

    probs = F.softmax(logits, dim=-1)
    top_prob, top_class_idx = torch.max(probs, dim=-1)

    confidence = top_prob.item()
    predicted_class_id = top_class_idx.item()
    predicted_class_label = bert_model.config.id2label[predicted_class_id]

    return {
        "insight_sectorial": predicted_class_label.replace("_", " ").title(),
        "confianza": confidence * 100
    }

def regex_keyword_extractor(text, regex_patterns, ontology):
    cleaned_text = clean_text(text)
    keywords, distribution = [], {func: 0 for func in ontology['enumerations']['FunctionsRSI']}
    for func, patterns in regex_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, cleaned_text)
            if matches:
                keywords.extend(matches)
                distribution[func] += len(matches)
    return {"palabras_clave": list(set(keywords)), "distribucion_funciones": distribution}
