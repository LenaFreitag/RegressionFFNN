async function run() {
  const statusElement = document.getElementById('status');

  if (statusElement) {
    statusElement.textContent = 'Gespeicherte Daten und Modelle werden geladen...';
  }

  const experimentData = await loadExperimentData();

  const trainData = experimentData.trainData;
  const testData = experimentData.testData;
  const noisyTrainData = experimentData.noisyTrainData;
  const noisyTestData = experimentData.noisyTestData;

  console.log('Experimentdaten geladen.');
  console.log('Trainingsdaten (clean):', trainData.length);
  console.log('Testdaten (clean):', testData.length);
  console.log('Trainingsdaten (noisy):', noisyTrainData.length);
  console.log('Testdaten (noisy):', noisyTestData.length);

  plotDataset(
    'plot-clean-data',
    trainData,
    testData,
    'Datensatz ohne Rauschen'
  );

  plotDataset(
    'plot-noisy-data',
    noisyTrainData,
    noisyTestData,
    'Datensatz mit Rauschen'
  );

  const cleanModel = await tf.loadLayersModel('models/clean-model/clean-model.json');
  const bestFitModel = await tf.loadLayersModel('models/best-fit-model/bestfit-model.json');
  const overfitModel = await tf.loadLayersModel('models/overfit-model/overfit-model.json');

  console.log('Modelle geladen.');

  const cleanTrainLoss = calculateMSE(cleanModel, trainData);
  const cleanTestLoss = calculateMSE(cleanModel, testData);

  const bestFitTrainLoss = calculateMSE(bestFitModel, noisyTrainData);
  const bestFitTestLoss = calculateMSE(bestFitModel, noisyTestData);

  const overfitTrainLoss = calculateMSE(overfitModel, noisyTrainData);
  const overfitTestLoss = calculateMSE(overfitModel, noisyTestData);

  console.log('Clean Model Train MSE:', cleanTrainLoss);
  console.log('Clean Model Test MSE:', cleanTestLoss);

  console.log('Best-Fit Model Train MSE:', bestFitTrainLoss);
  console.log('Best-Fit Model Test MSE:', bestFitTestLoss);

  console.log('Overfit Model Train MSE:', overfitTrainLoss);
  console.log('Overfit Model Test MSE:', overfitTestLoss);

  const cleanPredictionCurve = createPredictionCurve(cleanModel, -2, 2, 200);
  const bestFitPredictionCurve = createPredictionCurve(bestFitModel, -2, 2, 200);
  const overfitPredictionCurve = createPredictionCurve(overfitModel, -2, 2, 200);

  plotPrediction(
    'plot-clean-train-prediction',
    trainData,
    cleanPredictionCurve,
    'Vorhersage ohne Rauschen auf Trainingsdaten'
  );

  plotPrediction(
    'plot-clean-test-prediction',
    testData,
    cleanPredictionCurve,
    'Vorhersage ohne Rauschen auf Testdaten'
  );

  plotPrediction(
    'plot-bestfit-train-prediction',
    noisyTrainData,
    bestFitPredictionCurve,
    'Best-Fit-Vorhersage auf verrauschten Trainingsdaten'
  );

  plotPrediction(
    'plot-bestfit-test-prediction',
    noisyTestData,
    bestFitPredictionCurve,
    'Best-Fit-Vorhersage auf verrauschten Testdaten'
  );

  plotPrediction(
    'plot-overfit-train-prediction',
    noisyTrainData,
    overfitPredictionCurve,
    'Overfit-Vorhersage auf verrauschten Trainingsdaten'
  );

  plotPrediction(
    'plot-overfit-test-prediction',
    noisyTestData,
    overfitPredictionCurve,
    'Overfit-Vorhersage auf verrauschten Testdaten'
  );

  showMSE('mse-clean-train', 'Trainingsdaten ohne Rauschen', cleanTrainLoss);
  showMSE('mse-clean-test', 'Testdaten ohne Rauschen', cleanTestLoss);

  showMSE('mse-bestfit-train', 'Verrauschte Trainingsdaten Best-Fit', bestFitTrainLoss);
  showMSE('mse-bestfit-test', 'Verrauschte Testdaten Best-Fit', bestFitTestLoss);

  showMSE('mse-overfit-train', 'Verrauschte Trainingsdaten Overfit', overfitTrainLoss);
  showMSE('mse-overfit-test', 'Verrauschte Testdaten Overfit', overfitTestLoss);

  if (statusElement) {
    statusElement.textContent = 'Gespeicherte Daten und vortrainierte Modelle wurden geladen. Ergebnisse sind berechnet.';
  }
}

document.addEventListener('DOMContentLoaded', run);