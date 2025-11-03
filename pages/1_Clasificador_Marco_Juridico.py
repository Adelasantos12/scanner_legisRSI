import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import json
import io
import plotly.express as px

# --- Configuración de la Página ---
st.set_page_config(page_title="Clasificador de Marco Jurídico", page_icon="📋", layout="wide")
st.title("📋 Clasificador de Marco Jurídico Nacional")
st.write("Suba o pegue una lista de leyes (una por línea) para clasificarlas según su relevancia para el RSI.")

# --- Carga de Modelos y Datos ---
@st.cache_resource
def load_resources():
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    with open('data/sector_centroids.json', 'r', encoding='utf-8') as f:
        centroids = {sector: np.array(embedding) for sector, embedding in json.load(f).items()}
    return model, centroids

model, centroids = load_resources()

# --- Lógica de Clasificación ---
def classify_laws(law_list):
    # (Misma función que antes)
    results = []
    law_embeddings = model.encode(law_list, convert_to_tensor=True)
    for i, law_name in enumerate(law_list):
        embedding = law_embeddings[i]
        similarities = {sector: util.pytorch_cos_sim(embedding, center_emb).item() for sector, center_emb in centroids.items()}
        best_sector = max(similarities, key=similarities.get)
        best_score = similarities[best_sector]
        if best_score >= 0.7: priority = "Alta (relevante RSI)"
        elif 0.5 <= best_score < 0.7: priority = "Media (posible relación)"
        else: priority = "Baja (no RSI)"
        results.append({"ley": law_name, "sector_probable": best_sector, "score": best_score, "prioridad": priority})
    return pd.DataFrame(results)

# --- Interfaz de Usuario ---
input_method = st.radio("Método de entrada:", ("Pegar texto", "Subir archivo CSV/TXT"))
law_input = None
if input_method == "Pegar texto":
    law_input = st.text_area("Pegue la lista de leyes aquí (una por línea):", height=200)
else:
    uploaded_file = st.file_uploader("Suba un archivo (.csv o .txt)", type=["csv", "txt"])
    if uploaded_file: law_input = uploaded_file.getvalue().decode("utf-8")

if st.button("Clasificar Leyes"):
    if law_input:
        laws = [line.strip() for line in law_input.split('\n') if line.strip()]
        with st.spinner("Generando embeddings y clasificando..."):
            st.session_state['classified_laws'] = classify_laws(laws)
    else:
        st.warning("Por favor, ingrese una lista de leyes.")

# --- Visualización de Resultados ---
if 'classified_laws' in st.session_state:
    df_results = st.session_state['classified_laws']
    st.header("Resultados de la Clasificación")

    # --- NUEVAS VISUALIZACIONES ---
    st.subheader("Resumen del Marco Jurídico")

    # Detección de control de tabaco
    tobacco_laws = df_results[df_results['ley'].str.contains("tabaco", case=False)]
    if not tobacco_laws.empty:
        st.info(f"🚭 Se ha detectado {len(tobacco_laws)} ley(es) que podría(n) estar relacionada(s) con el control de tabaco.")
        st.dataframe(tobacco_laws, hide_index=True)

    # Resumen cuantitativo
    relevant_laws_count = len(df_results[df_results['prioridad'].isin(["Alta (relevante RSI)", "Media (posible relación)"])])
    st.metric(label="Leyes con Posible Relevancia RSI", value=f"{relevant_laws_count} de {len(df_results)}")

    # Gráfico de barras
    sector_counts = df_results['sector_probable'].value_counts().reset_index()
    sector_counts.columns = ['Sector', 'Número de Leyes']
    fig = px.bar(sector_counts, x='Número de Leyes', y='Sector', orientation='h', title='Distribución de Leyes por Sector RSI Probable')
    st.plotly_chart(fig, use_container_width=True)
    # --- FIN DE NUEVAS VISUALIZACIONES ---

    st.subheader("Detalle y Selección para Escaneo")
    df_display = df_results.copy()
    df_display['seleccionar'] = True

    # Filtros y Tabla
    col1, col2 = st.columns(2)
    with col1: priority_filter = st.multoselect("Filtrar por prioridad:", options=df_display['prioridad'].unique(), default=df_display['prioridad'].unique())
    with col2: sector_filter = st.multoselect("Filtrar por sector:", options=df_display['sector_probable'].unique(), default=df_display['sector_probable'].unique())
    filtered_df = df_display[(df_display['prioridad'].isin(priority_filter)) & (df_display['sector_probable'].isin(sector_filter))]

    edited_df = st.data_editor(filtered_df, column_config={"ley": st.column_config.TextColumn("Ley", width="large"), "sector_probable": st.column_config.TextColumn("Sector Probable"), "score": st.column_config.ProgressColumn("Score", format="%.2f", min_value=0.0, max_value=1.0), "prioridad": st.column_config.TextColumn("Prioridad"), "seleccionar": st.column_config.CheckboxColumn("Seleccionar", default=True)}, hide_index=True, key='data_editor')

    selected_laws = edited_df[edited_df['seleccionar']]

    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"Enviar {len(selected_laws)} leyes a Escaneo RSI"):
            st.session_state['laws_to_scan'] = selected_laws['ley'].tolist()
            st.success(f"{len(selected_laws)} leyes enviadas al módulo de escaneo. Navegue a la página principal para verlas.")
    with col2:
        output = io.BytesIO()
        selected_laws.to_csv(output, index=False, encoding='utf-8')
        st.download_button(label="Descargar Selección como CSV", data=output.getvalue(), file_name='leyes_clasificadas_seleccion.csv', mime='text/csv')
