// src/models/WineModel.js

class WineModel {
    constructor() {
        this.features = [
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol"
        ];

        this.initializeWeights();
    }

    initializeWeights() {
        this.weights = {};
        this.features.forEach(feature => {
            this.weights[feature] = Math.random() * 0.1;
        });
        this.bias = Math.random() * 0.1;
    }

    predict(wine) {
        let sum = this.bias;
        for (let feature of this.features) {
            sum += this.weights[feature] * wine[feature];
        }
        return this.sigmoid(sum);
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
}

module.exports = WineModel;