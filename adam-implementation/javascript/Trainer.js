class Trainer {
    constructor(model, learningRate = 0.01, useAdam = false, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
        this.model = model;
        this.learningRate = learningRate;
        this.useAdam = useAdam;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.epochs = 100;
        this.batchSize = 32;

        // Inicializar historial de pérdida y precisión
        this.history = {
            loss: [],
            accuracy: []
        };

        // Inicializar momentos si se usa Adam
        if (useAdam) {
            this.m = {};
            this.v = {};
            for (let feature of Object.keys(this.model.weights)) {
                this.m[feature] = 0;
                this.v[feature] = 0;
            }
            this.m.bias = 0;
            this.v.bias = 0;
        }
    }

    calculateLoss(predicted, actual, lambda = 0.01, regularizationType = "L2") {
        const epsilon = 1e-15;
        predicted = Math.min(Math.max(predicted, epsilon), 1 - epsilon);
        
        // Binary Cross-Entropy Loss
        let crossEntropyLoss = -(actual * Math.log(predicted) + (1 - actual) * Math.log(1 - predicted));
    
        // Regularización
        let regularizationTerm = 0;
        if (regularizationType === "L2") {
            // L2: Penaliza la suma de los cuadrados de los pesos
            regularizationTerm = lambda * Object.values(this.model.weights).reduce((sum, w) => sum + w ** 2, 0);
        } else if (regularizationType === "L1") {
            // L1: Penaliza la suma absoluta de los pesos
            regularizationTerm = lambda * Object.values(this.model.weights).reduce((sum, w) => sum + Math.abs(w), 0);
        } else if (regularizationType === "ElasticNet") {
            // Elastic Net: Combina L1 y L2
            let l1 = Object.values(this.model.weights).reduce((sum, w) => sum + Math.abs(w), 0);
            let l2 = Object.values(this.model.weights).reduce((sum, w) => sum + w ** 2, 0);
            regularizationTerm = lambda * (0.5 * l1 + 0.5 * l2);
        }
    
        return crossEntropyLoss + regularizationTerm;
    }
    

    calculateGradients(wine, predicted, actual) {
        const error = predicted - actual;
        const gradients = {};

        for (let feature in wine) {
            gradients[feature] = error * wine[feature];
        }

        gradients.bias = error;
        return gradients;
    }

    updateWeights(gradients, t) {
        for (let feature in this.model.weights) {
            if (this.useAdam) {
                // Adam: Cálculo de momentos de primer y segundo orden
                this.m[feature] = this.beta1 * this.m[feature] + (1 - this.beta1) * gradients[feature];
                this.v[feature] = this.beta2 * this.v[feature] + (1 - this.beta2) * (gradients[feature] ** 2);

                // Corrección de sesgo
                let m_hat = this.m[feature] / (1 - Math.pow(this.beta1, t));
                let v_hat = this.v[feature] / (1 - Math.pow(this.beta2, t));

                // Actualización del peso usando Adam
                this.model.weights[feature] -= this.learningRate * m_hat / (Math.sqrt(v_hat) + this.epsilon);
            } else {
                // Stochastic Gradient Descent: Actualización estándar del peso
                this.model.weights[feature] -= this.learningRate * gradients[feature];
            }
        }

        // Actualización del bias con Adam o Stochastic Gradient Descent
        if (this.useAdam) {
            this.m.bias = this.beta1 * this.m.bias + (1 - this.beta1) * gradients.bias;
            this.v.bias = this.beta2 * this.v.bias + (1 - this.beta2) * (gradients.bias ** 2);

            let m_hat_bias = this.m.bias / (1 - Math.pow(this.beta1, t));
            let v_hat_bias = this.v.bias / (1 - Math.pow(this.beta2, t));

            this.model.bias -= this.learningRate * m_hat_bias / (Math.sqrt(v_hat_bias) + this.epsilon);
        } else {
            this.model.bias -= this.learningRate * gradients.bias;
        }
    }

    train(trainData) {
        let t = 0;
        let method = this.useAdam ? "Adam" : "SGD";
        console.log(`🔹 Entrenando con ${method}...`);

        for (let epoch = 0; epoch < this.epochs; epoch++) {
            let totalLoss = 0;
            let correctPredictions = 0;
            const shuffledData = [...trainData].sort(() => Math.random() - 0.5);

            for (let i = 0; i < shuffledData.length; i += this.batchSize) {
                const batch = shuffledData.slice(i, i + this.batchSize);
                let batchGradients = {};
                let batchLoss = 0;

                batch.forEach(example => {
                    const predicted = this.model.predict(example.features);
                    const loss = this.calculateLoss(predicted, example.quality);
                    batchLoss += loss;

                    if (Math.round(predicted) === example.quality) {
                        correctPredictions++;
                    }

                    const gradients = this.calculateGradients(example.features, predicted, example.quality);

                    for (let feature in gradients) {
                        batchGradients[feature] = (batchGradients[feature] || 0) + gradients[feature] / batch.length;
                    }
                });

                t++;
                this.updateWeights(batchGradients, t);
                totalLoss += batchLoss / batch.length;
            }

            const accuracy = correctPredictions / trainData.length;
            const averageLoss = totalLoss / Math.ceil(trainData.length / this.batchSize);

            this.history.loss.push(averageLoss);
            this.history.accuracy.push(accuracy);

            if ((epoch + 1) % 10 === 0) {
                console.log(`Época ${epoch + 1}/${this.epochs} (${method})`);
                console.log(`  Pérdida: ${averageLoss.toFixed(4)}`);
                console.log(`  Precisión: ${(accuracy * 100).toFixed(2)}%`);
            }
        }
    }

    evaluate(testData) {
        let correctPredictions = 0;
        let totalLoss = 0;

        testData.forEach(example => {
            const predicted = this.model.predict(example.features);
            const loss = this.calculateLoss(predicted, example.quality);

            if (Math.round(predicted) === example.quality) {
                correctPredictions++;
            }
            totalLoss += loss;
        });

        return {
            accuracy: correctPredictions / testData.length,
            averageLoss: totalLoss / testData.length
        };
    }
}

module.exports = Trainer;
