const QuickChart = require('quickchart-js');
const fs = require('fs');

/**
 * Generates a chart URL for loss and accuracy history.
 * @param {Object} history - The training history containing loss and accuracy.
 * @param {string} title - The title of the chart.
 * @param {string} filename - The file to save the chart URL.
 */
function generateChart(history, title, filename) {
    const chart = new QuickChart();
    chart.setConfig({
        type: 'line',
        data: {
            labels: Array.from({ length: history.loss.length }, (_, i) => i + 1),
            datasets: [
                {
                    label: 'Pérdida',
                    data: history.loss,
                    borderColor: 'red',
                    fill: false
                },
                {
                    label: 'Precisión',
                    data: history.accuracy.map(acc => acc * 100),
                    borderColor: 'blue',
                    fill: false
                }
            ]
        },
        options: {
            title: {
                display: true,
                text: title
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    const chartUrl = chart.getUrl();
    console.log(`\n📊 Gráfico generado para ${title}: ${chartUrl}`);

    fs.writeFileSync(filename, chartUrl);
}

module.exports = generateChart;
