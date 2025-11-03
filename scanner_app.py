import streamlit as st
from core.analysis import load_analysis_resources, legal_metadata_resolver, semantic_analyzer, regex_keyword_extractor
import pandas as pd
import plotly.express as px
import io

# --- Configuración de Página y Carga de Recursos ---
st.set_page_config(page_title="Scanner Legis-RSI", page_icon="⚖️", layout="wide")

@st.cache_resource
def cached_load_resources():
    return load_analysis_resources()

bert_tokenizer, bert_model, ontology, regex_patterns, legal_catalog = cached_load_resources()

st.title("⚖️ Módulo de Escaneo Profundo de Leyes")
st.sidebar.success("Seleccione un módulo.")

# --- Estado de Sesión y Comunicación entre Páginas ---
if 'laws_to_scan' not in st.session_state:
    st.session_state['laws_to_scan'] = []

# --- Interfaz de Usuario ---
st.header("Entrada de Texto")

# Opción para usar leyes del clasificador
if st.session_state['laws_to_scan']:
    selected_law = st.selectbox(
        "Seleccione una ley pre-clasificada para un análisis profundo:",
        options=st.session_state['laws_to_scan'],
        index=0
    )
    st.info("Para analizar un texto nuevo, primero elimine la selección de arriba.")
    text_input = selected_law
else:
    text_input = st.text_area("O pegue el texto completo de una ley aquí:", height=250)

# Opción para subir archivo (deshabilitada si se usa texto del clasificador)
uploaded_file = st.file_uploader(
    "O suba un archivo de texto (.txt):",
    type=['txt'],
    disabled=bool(st.session_state['laws_to_scan'])
)

if uploaded_file:
    text_input = uploaded_file.getvalue().decode("utf-8")

# --- Botón de Análisis y Lógica ---
if st.button("Realizar Escaneo Profundo"):
    if text_input and text_input.strip():
        with st.spinner("Realizando análisis semántico profundo con modelo BERT..."):

            # Ejecutar el pipeline de análisis
            legal_meta = legal_metadata_resolver(text_input, legal_catalog)
            semantic_results = semantic_analyzer(text_input, bert_tokenizer, bert_model)
            regex_results = regex_keyword_extractor(text_input, regex_patterns, ontology)

            # Guardar resultados para visualización
            st.session_state['analysis_results'] = {
                **legal_meta,
                **semantic_results,
                **regex_results
            }
    else:
        st.warning("Por favor, ingrese un texto o seleccione una ley para analizar.")

# --- Visualización de Resultados ---
if 'analysis_results' in st.session_state and st.session_state['analysis_results']:
    results = st.session_state['analysis_results']

    st.header("Resultados del Análisis")

    # Métricas principales
    col1, col2, col3 = st.columns(3)
    col1.metric("Ley Analizada", results['ley_analizada'])
    col2.metric("Insight Sectorial (BERT)", results['insight_sectorial'])
    col3.metric("Confianza del Modelo", f"{results['confianza']:.2f}%")

    # Pestañas de detalles
    tab1, tab2 = st.tabs(["📊 Gráfico de Distribución (Regex)", "🔑 Coincidencias Regex"])

    with tab1:
        dist_df = pd.DataFrame(
            list(results['distribucion_funciones'].items()),
            columns=['Función', 'Coincidencias']
        ).sort_values(by='Coincidencias', ascending=False)

        fig = px.bar(dist_df, x='Coincidencias', y='Función', orientation='h', title='Distribución de Funciones RSI (por Regex)')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if results['palabras_clave']:
            st.write(results['palabras_clave'])
        else:
            st.info("No se encontraron coincidencias directas con expresiones regulares.")

    # Expander para ver el texto analizado
    with st.expander("Ver texto original analizado"):
        st.text(text_input)
