function groundTruth(x) {
  return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

function randomUniform(min, max) {
  return Math.random() * (max - min) + min;
}

function generateDataset(n, min, max) {
  const data = [];

  for (let i = 0; i < n; i++) {
    const x = randomUniform(min, max);
    const y = groundTruth(x);

    data.push({
      x: x,
      y: y
    });
  }

  return data;
}

function shuffleArray(array) {
  const shuffled = [...array];

  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  return shuffled;
}

function splitDataset(data, trainSize) {
  const shuffledData = shuffleArray(data);

  const trainData = shuffledData.slice(0, trainSize);
  const testData = shuffledData.slice(trainSize);

  return {
    trainData: trainData,
    testData: testData
  };
}

function addNoise(data, variance) {
  const standardDeviation = Math.sqrt(variance);

  const noiseTensor = tf.randomNormal([data.length], 0, standardDeviation);
  const noiseValues = noiseTensor.arraySync();

  noiseTensor.dispose();

  return data.map((point, index) => {
    return {
      x: point.x,
      y: point.y + noiseValues[index]
    };
  });
}

function downloadJSON(data, filename) {
  const jsonString = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonString], {
    type: 'application/json'
  });

  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();

  URL.revokeObjectURL(url);
}

async function loadExperimentData() {
  const response = await fetch('data/experiment-data.json');

  if (!response.ok) {
    throw new Error('experiment-data.json konnte nicht geladen werden.');
  }

  return await response.json();
}