import os
import re
import json
import pandas as pd
import yaml
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename
import docx
from PyPDF2 import PdfReader
import io
import csv
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch.nn.functional as F

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Carga de Recursos ---
with open('data/ontology_rsi_mexico.json', 'r', encoding='utf-8') as f:
    ontology = json.load(f)
with open('data/regex_patterns_mexico.yaml', 'r', encoding='utf-8') as f:
    regex_patterns = yaml.safe_load(f)
with open('data/legal_metadata_catalog.json', 'r', encoding='utf-8') as f:
    legal_catalog = json.load(f)

MODEL_DIR = "models/bert_rsi_latam"
tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- Lógica de Análisis ---
def clean_text(text):
    text = text.lower()
    return re.sub(r'\s+', ' ', text)

def legal_metadata_resolver(text):
    cleaned_text = clean_text(text)
    for key, metadata in legal_catalog.items():
        if re.search(r'\b' + re.escape(metadata['nombre'].lower()) + r'\b', cleaned_text) or \
           re.search(r'\b' + re.escape(metadata['sigla_local'].lower()) + r'\b', cleaned_text):
            return {"ley_analizada": f"{metadata['nombre']} ({metadata['autoridad_responsable_principal']['sigla_local']})", "autoridad_legal": metadata['autoridad_responsable_principal']['nombre_local'], "dof": metadata.get('dof_publicacion', 'No disponible')}
    return {"ley_analizada": "No identificada", "autoridad_legal": "No identificada", "dof": "No disponible"}

def semantic_analyzer(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    top_prob, top_class_idx = torch.max(probs, dim=-1)
    confidence = top_prob.item()
    predicted_class_id = top_class_idx.item()
    predicted_class_label = model.config.id2label[predicted_class_id]
    return {"insight_sectorial": predicted_class_label.replace("_", " ").title(), "confianza": confidence * 100}

def regex_keyword_extractor(text):
    cleaned_text = clean_text(text)
    keywords, distribution = [], {func: 0 for func in ontology['enumerations']['FunctionsRSI']}
    for func, patterns in regex_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, cleaned_text)
            if matches:
                keywords.extend(matches)
                distribution[func] += len(matches)
    return {"palabras_clave": list(set(keywords)), "distribucion_funciones": distribution}

# --- Lectura de Archivos ---
def read_file(file):
    if file.filename.endswith('.txt'): return file.read().decode('utf-8')
    elif file.filename.endswith('.docx'):
        doc = docx.Document(io.BytesIO(file.read()))
        return ' '.join([para.text for para in doc.paragraphs])
    elif file.filename.endswith('.pdf'):
        pdf = PdfReader(io.BytesIO(file.read()))
        return ' '.join([page.extract_text() for page in pdf.pages])
    return ''

# --- Endpoints de la API ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    text = ''
    if 'text' in request.form and request.form['text']: text = request.form['text']
    elif 'file' in request.files and request.files['file'].filename: text = read_file(request.files['file'])
    if not text: return jsonify({'error': 'No text or file provided'}), 400

    final_results = {**legal_metadata_resolver(text), **semantic_analyzer(text), **regex_keyword_extractor(text)}
    return jsonify(final_results)

@app.route('/download_csv', methods=['POST'])
def download_csv():
    data = request.json
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Metrica', 'Valor'])
    writer.writerow(['Ley Analizada', data.get('ley_analizada')])
    writer.writerow(['Autoridad Legal', data.get('autoridad_legal')])
    writer.writerow(['Publicación (DOF)', data.get('dof')])
    writer.writerow(['Insight Sectorial Dominante', data.get('insight_sectorial')])
    writer.writerow(['Confianza del Modelo (%)', f"{data.get('confianza', 0):.2f}"])
    writer.writerow(['Palabras Clave (Regex)', ', '.join(data.get('palabras_clave', []))])
    output.seek(0)
    return Response(output, mimetype="text/csv", headers={"Content-Disposition":"attachment;filename=analisis_rsi.csv"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
