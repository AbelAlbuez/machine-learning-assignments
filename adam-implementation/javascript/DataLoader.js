// src/utils/DataLoader.js
const fs = require('fs');
const csv = require('csv-parser');
const Statistics = require('./Statistics');

class DataLoader {
    constructor() {
        this.statistics = new Statistics();
        this.rawData = [];
        this.normalizedData = [];
    }

    /**
     * Carga los datos del archivo CSV
     * @param {string} filePath - Ruta al archivo CSV
     * @returns {Promise} Promesa con los datos cargados
     */
    loadData(filePath) {
        return new Promise((resolve, reject) => {
            const results = [];

            fs.createReadStream(filePath)
                .pipe(csv())
                .on('data', (data) => {
                    // Convertimos los strings a números
                    const processedWine = {
                        features: {
                            "fixed acidity": parseFloat(data["fixed acidity"]),
                            "volatile acidity": parseFloat(data["volatile acidity"]),
                            "citric acid": parseFloat(data["citric acid"]),
                            "residual sugar": parseFloat(data["residual sugar"]),
                            "chlorides": parseFloat(data["chlorides"]),
                            "free sulfur dioxide": parseFloat(data["free sulfur dioxide"]),
                            "total sulfur dioxide": parseFloat(data["total sulfur dioxide"]),
                            "density": parseFloat(data["density"]),
                            "pH": parseFloat(data["pH"]),
                            "sulphates": parseFloat(data["sulphates"]),
                            "alcohol": parseFloat(data["alcohol"])
                        },
                        quality: parseFloat(data.quality) > 6.5 ? 1 : 0
                    };
                    results.push(processedWine);
                })
                .on('end', () => {
                    this.rawData = results;
                    this.processData();
                    resolve(this.normalizedData);
                })
                .on('error', (error) => reject(error));
        });
    }

    /**
     * Procesa y normaliza los datos cargados
     */
    processData() {
        // Calculamos estadísticas para cada característica
        const features = Object.keys(this.rawData[0].features);
        
        features.forEach(feature => {
            const values = this.rawData.map(wine => wine.features[feature]);
            this.statistics.calculateFeatureStats(feature, values);
        });

        // Normalizamos los datos
        this.normalizedData = this.rawData.map(wine => ({
            features: this.normalizeWine(wine.features),
            quality: wine.quality
        }));
    }

    /**
     * Normaliza las características de un vino
     * @param {Object} wine - Características del vino
     * @returns {Object} Características normalizadas
     */
    normalizeWine(wine) {
        const normalizedFeatures = {};
        
        for (let feature in wine) {
            const stats = this.statistics.getStats(feature);
            normalizedFeatures[feature] = 
                (wine[feature] - stats.mean) / stats.stdDev;
        }
        
        return normalizedFeatures;
    }

    /**
     * Divide los datos en conjuntos de entrenamiento y prueba
     * @param {number} trainRatio - Proporción para entrenamiento (0-1)
     * @returns {Object} Datos divididos
     */
    splitData(trainRatio = 0.8) {
        const totalSamples = this.normalizedData.length;
        const trainSize = Math.floor(totalSamples * trainRatio);
        
        // Mezclamos los datos aleatoriamente
        const shuffledData = [...this.normalizedData]
            .sort(() => Math.random() - 0.5);

        return {
            trainData: shuffledData.slice(0, trainSize),
            testData: shuffledData.slice(trainSize)
        };
    }

    /**
     * Obtiene estadísticas del dataset
     * @returns {Object} Resumen estadístico
     */
    getDataSummary() {
        return {
            totalSamples: this.rawData.length,
            goodWines: this.rawData.filter(w => w.quality === 1).length,
            regularWines: this.rawData.filter(w => w.quality === 0).length,
            statistics: this.statistics.getAllStats()
        };
    }
}

module.exports = DataLoader;