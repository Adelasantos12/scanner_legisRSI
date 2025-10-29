document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const textInput = document.getElementById('text-input');
    const fileInput = document.getElementById('file-input');
    const resultsSection = document.getElementById('results-section');
    let chart;
    let analysisData;

    analyzeBtn.addEventListener('click', async () => {
        const formData = new FormData();
        if (fileInput.files.length > 0) {
            formData.append('file', fileInput.files[0]);
        } else {
            formData.append('text', textInput.value);
        }

        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            analysisData = await response.json();
            displayResults(analysisData);
        } else {
            alert('Error al analizar el texto.');
        }
    });

    function displayResults(data) {
        document.getElementById('rsi-function').textContent = data.funcion_rsi;
        document.getElementById('sector').textContent = data.sector;
        document.getElementById('confidence').textContent = data.confianza.toFixed(2);
        document.getElementById('legal-authority').textContent = data.autoridad_legal;
        document.getElementById('probable-law').textContent = data.ley_probable;

        const keywordsContainer = document.getElementById('keywords');
        keywordsContainer.innerHTML = data.palabras_clave.map(kw => `<span>${kw}</span>`).join(', ');

        const explanationContainer = document.getElementById('explanation');
        explanationContainer.innerHTML = '';
        for(const func in data.explicacion){
            if (data.explicacion[func].length > 0){
                const funcTitle = document.createElement('h4');
                funcTitle.textContent = func;
                explanationContainer.appendChild(funcTitle);
                data.explicacion[func].forEach(exp => {
                    const p = document.createElement('p');
                    p.textContent = exp;
                    explanationContainer.appendChild(p);
                });
            }
        }

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
                    label: 'Distribución de Funciones RSI',
                    data: Object.values(distribution),
                    backgroundColor: 'rgba(0, 123, 255, 0.2)',
                    borderColor: 'rgba(0, 123, 255, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    r: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    document.getElementById('download-csv-btn').addEventListener('click', async () => {
        const response = await fetch('/download_csv', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
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
