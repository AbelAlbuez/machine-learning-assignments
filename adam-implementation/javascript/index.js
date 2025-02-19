// src/index.js
const WineModel = require('./WineModel');
const DataLoader = require('./DataLoader');
const Trainer = require('./Trainer');
const generateChart = require('./generateChart');




async function main() {
    try {
        // 1. Cargamos y preparamos los datos
        console.log('Cargando datos...');
        const dataLoader = new DataLoader();
        await dataLoader.loadData('../datasets/winequality-red.csv');

        // 2. Dividimos los datos en entrenamiento y prueba
        const { trainData, testData } = dataLoader.splitData(0.8); // 80% entrenamiento, 20% prueba
        console.log(`Datos divididos en: ${trainData.length} entrenamiento, ${testData.length} prueba`);

        // 3. Entrenamiento con SGD (Sin Adam)
        console.log('\nEntrenando con SGD (Descenso de Gradiente Estocástico)...');
        const modelSGD = new WineModel();
        const trainerSGD = new Trainer(modelSGD, 0.01, false); // SGD sin Adam
        trainerSGD.train(trainData);
        const resultsSGD = trainerSGD.evaluate(testData);
        console.log(`\n🔸 Resultado con SGD:`);
        console.log(`   Precisión: ${(resultsSGD.accuracy * 100).toFixed(2)}%`);
        console.log(`   Pérdida: ${resultsSGD.averageLoss.toFixed(4)}`);

        // 4. Entrenamiento con Adam
        console.log('\nEntrenando con Adam...');
        const modelAdam = new WineModel();
        const trainerAdam = new Trainer(modelAdam, 0.01, true); // Adam activado
        trainerAdam.train(trainData);

        const resultsAdam = trainerAdam.evaluate(testData);
        console.log(`\n🔹 Resultado con Adam:`);
        console.log(`   Precisión: ${(resultsAdam.accuracy * 100).toFixed(2)}%`);
        console.log(`   Pérdida: ${resultsAdam.averageLoss.toFixed(4)}`);

        // 5. Probamos con algunos ejemplos
        console.log('\nProbando predicciones con Adam...');
        //const exampleWines = testData.slice(0, 5);
        let good = 0;
        testData.forEach((wine, index) => {
            const prediction = modelAdam.predict(wine.features);
            // console.log(`\nVino ${index + 1}:`);
            // console.log(`Predicción: ${prediction.toFixed(4)} (${prediction > 0.5 ? 'Bueno' : 'Regular'})`);
            // console.log(`Realidad: ${wine.quality} (${wine.quality === 1 ? 'Bueno' : 'Regular'})`);
            if (wine.quality === 1 && prediction > 0.5) {
                good += 1
            }

            if (wine.quality !== 1 && prediction < 0.5) {
                good += 1
            }
        });

        console.log(`tasa de acierto: ${(good / testData.length) * 100}`)
        generateChart(trainerSGD.history, "SGD - Pérdida y Precisión", "sgd_chart_url.txt");
        generateChart(trainerAdam.history, "Adam - Pérdida y Precisión", "adam_chart_url.txt");
    } catch (error) {
        console.error('Error:', error);
    }
}

// Ejecutamos
main();
