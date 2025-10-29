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

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Cargar ontología, patrones de regex y datos de entrenamiento
with open('data/ontology_rsi_mexico.json', 'r', encoding='utf-8') as f:
    ontology = json.load(f)
with open('data/regex_patterns_mexico.yaml', 'r', encoding='utf-8') as f:
    regex_patterns = yaml.safe_load(f)
training_data = pd.read_csv('data/training_data_mexico_extended.csv')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

def analyze_text(text):
    cleaned_text = clean_text(text)
    results = {
        'funcion_rsi': 'No identificada',
        'sector': 'No identificado',
        'confianza': 0,
        'palabras_clave': [],
        'autoridad_legal': 'No identificada',
        'ley_probable': 'No identificada',
        'distribucion_funciones': {func: 0 for func in ontology['enumerations']['FunctionsRSI']},
        'explicacion': {}
    }

    # Lógica de clasificación basada en regex
    for func, patterns in regex_patterns.items():
        results['explicacion'][func] = []
        for pattern in patterns:
            for match in re.finditer(pattern, cleaned_text):
                results['distribucion_funciones'][func] += 1
                results['palabras_clave'].append(match.group(0))

                # Guardar el fragmento de texto para la explicación
                context_window = cleaned_text[max(0, match.start() - 50):min(len(cleaned_text), match.end() + 50)]
                results['explicacion'][func].append(f"...{context_window}...")


    # Simular la función y confianza principal
    total_matches = sum(results['distribucion_funciones'].values())
    if total_matches > 0:
        main_func = max(results['distribucion_funciones'], key=results['distribucion_funciones'].get)
        results['funcion_rsi'] = main_func
        results['confianza'] = (results['distribucion_funciones'][main_func] / total_matches) * 100

        # Inferir sector, autoridad y ley a partir de los datos de entrenamiento
        relevant_data = training_data[training_data['funcion'] == main_func]
        if not relevant_data.empty:
            results['sector'] = relevant_data['sector'].iloc[0]
            results['autoridad_legal'] = relevant_data['actor'].iloc[0]
            results['ley_probable'] = relevant_data['instrumento'].iloc[0]

    return results

def read_file(file):
    if file.filename.endswith('.txt'):
        return file.read().decode('utf-8')
    elif file.filename.endswith('.docx'):
        doc = docx.Document(io.BytesIO(file.read()))
        return ' '.join([para.text for para in doc.paragraphs])
    elif file.filename.endswith('.pdf'):
        pdf = PdfReader(io.BytesIO(file.read()))
        return ' '.join([page.extract_text() for page in pdf.pages])
    return ''

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    text = ''
    if 'text' in request.form and request.form['text']:
        text = request.form['text']
    elif 'file' in request.files and request.files['file'].filename:
        file = request.files['file']
        text = read_file(file)

    if not text:
        return jsonify({'error': 'No text or file provided'}), 400

    analysis_results = analyze_text(text)

    # Guardar resultados en log
    with open('logs/results.log', 'a', encoding='utf-8') as f:
        log_entry = {'input_text': text, 'results': analysis_results}
        f.write(json.dumps(log_entry) + '\n')

    return jsonify(analysis_results)

@app.route('/download_csv', methods=['POST'])
def download_csv():
    data = request.json
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(['Metrica', 'Valor'])
    writer.writerow(['Función RSI', data.get('funcion_rsi')])
    writer.writerow(['Sector', data.get('sector')])
    writer.writerow(['Confianza (%)', data.get('confianza')])
    writer.writerow(['Autoridad Legal', data.get('autoridad_legal')])
    writer.writerow(['Ley Probable', data.get('ley_probable')])
    writer.writerow(['Palabras Clave', ', '.join(data.get('palabras_clave', []))])

    output.seek(0)
    return Response(output, mimetype="text/csv", headers={"Content-Disposition":"attachment;filename=analisis_rsi.csv"})


if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
