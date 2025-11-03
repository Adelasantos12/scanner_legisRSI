document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const textInput = document.getElementById('text-input');
    const fileInput = document.getElementById('file-input');
    const resultsSection = document.getElementById('results-section');
    const loader = document.getElementById('loader');
    let chart;
    let analysisData;

    analyzeBtn.addEventListener('click', async () => {
        const formData = new FormData();
        if (fileInput.files.length > 0) {
            formData.append('file', fileInput.files[0]);
        } else if (textInput.value.trim()) {
            formData.append('text', textInput.value);
        } else {
            alert('Por favor, ingrese texto o seleccione un archivo.');
            return;
        }

        // --- Inicia el estado de carga ---
        analyzeBtn.disabled = true;
        loader.style.display = 'flex';
        resultsSection.style.display = 'none';

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                analysisData = await response.json();
                displayResults(analysisData);
            } else {
                alert('Error al analizar el texto. Por favor, intente de nuevo.');
            }
        } catch (error) {
            console.error('Error en la solicitud:', error);
            alert('Ocurrió un error de red. Por favor, verifique su conexión e intente de nuevo.');
        } finally {
            // --- Finaliza el estado de carga ---
            analyzeBtn.disabled = false;
            loader.style.display = 'none';
        }
    });

    function displayResults(data) {
        document.getElementById('ley-analizada').textContent = data.ley_analizada;
        document.getElementById('autoridad-legal').textContent = data.autoridad_legal;
        document.getElementById('dof').textContent = data.dof;
        document.getElementById('insight-sectorial').textContent = data.insight_sectorial;
        document.getElementById('confianza').textContent = data.confianza.toFixed(2);

        const keywordsContainer = document.getElementById('keywords');
        keywordsContainer.innerHTML = data.palabras_clave.length > 0
            ? data.palabras_clave.map(kw => `<span>${kw}</span>`).join(', ')
            : 'No se encontraron coincidencias.';

        renderChart(data.distribucion_funciones);
        resultsSection.style.display = 'block';
    }

    function renderChart(distribution) {
        const ctx = document.getElementById('rsi-chart').getContext('2d');
        if (chart) {
            chart.destroy();
        }
        chart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: Object.keys(distribution),
                datasets: [{
                    label: 'Distribución de Funciones RSI por Regex',
                    data: Object.values(distribution),
                    backgroundColor: 'rgba(0, 123, 255, 0.2)',
                    borderColor: 'rgba(0, 123, 255, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    r: { beginAtZero: true }
                }
            }
        });
    }

    document.getElementById('download-csv-btn').addEventListener('click', async () => {
        if (!analysisData) {
            alert("Primero debe realizar un análisis.");
            return;
        }
        const response = await fetch('/download_csv', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(analysisData)
        });

        if(response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = 'analisis_rsi.csv';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
        } else {
            alert('Error al descargar el CSV.');
        }
    });
});
