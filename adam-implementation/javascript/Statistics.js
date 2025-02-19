// src/utils/Statistics.js

class Statistics {
    constructor() {
        // Almacenará las estadísticas de cada característica
        this.stats = {};
    }

    /**
     * Calcula la media de un conjunto de valores
     * @param {Array<number>} values - Array de números
     * @returns {number} Media aritmética
     */
    calculateMean(values) {
        const sum = values.reduce((acc, val) => acc + val, 0);
        return sum / values.length;
    }

    /**
     * Calcula la desviación estándar
     * @param {Array<number>} values - Array de números
     * @param {number} mean - Media de los valores
     * @returns {number} Desviación estándar
     */
    calculateStdDev(values, mean) {
        const squaredDiffs = values.map(value => 
            Math.pow(value - mean, 2)
        );
        const variance = this.calculateMean(squaredDiffs);
        return Math.sqrt(variance);
    }

    /**
     * Calcula estadísticas para una característica
     * @param {string} feature - Nombre de la característica
     * @param {Array<number>} values - Valores de la característica
     */
    calculateFeatureStats(feature, values) {
        const mean = this.calculateMean(values);
        const stdDev = this.calculateStdDev(values, mean);
        
        this.stats[feature] = {
            mean,
            stdDev,
            min: Math.min(...values),
            max: Math.max(...values),
            count: values.length,
            // Calculamos percentiles para entender la distribución
            percentiles: this.calculatePercentiles(values)
        };
    }

    /**
     * Calcula percentiles importantes
     * @param {Array<number>} values - Array de números
     * @returns {Object} Objeto con percentiles
     */
    calculatePercentiles(values) {
        const sorted = [...values].sort((a, b) => a - b);
        return {
            p25: sorted[Math.floor(values.length * 0.25)],
            p50: sorted[Math.floor(values.length * 0.50)], // mediana
            p75: sorted[Math.floor(values.length * 0.75)]
        };
    }

    /**
     * Obtiene las estadísticas de una característica
     * @param {string} feature - Nombre de la característica
     * @returns {Object} Estadísticas de la característica
     */
    getStats(feature) {
        return this.stats[feature];
    }

    /**
     * Obtiene todas las estadísticas
     * @returns {Object} Todas las estadísticas
     */
    getAllStats() {
        return this.stats;
    }

    /**
     * Genera un resumen de las estadísticas
     * @returns {string} Resumen formateado
     */
    generateSummary() {
        let summary = "=== Resumen Estadístico ===\n\n";
        
        for (let feature in this.stats) {
            summary += `${feature}:\n`;
            summary += `  Media: ${this.stats[feature].mean.toFixed(4)}\n`;
            summary += `  Desv. Est.: ${this.stats[feature].stdDev.toFixed(4)}\n`;
            summary += `  Min: ${this.stats[feature].min.toFixed(4)}\n`;
            summary += `  Max: ${this.stats[feature].max.toFixed(4)}\n`;
            summary += `  Mediana: ${this.stats[feature].percentiles.p50.toFixed(4)}\n`;
            summary += '\n';
        }

        return summary;
    }
}

module.exports = Statistics;