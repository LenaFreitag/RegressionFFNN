function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.dense({
    inputShape: [1],
    units: 100,
    activation: 'relu'
  }));

  model.add(tf.layers.dense({
    units: 100,
    activation: 'relu'
  }));

  model.add(tf.layers.dense({
    units: 1
  }));

  return model;
}

function createLargeModel() {
  const model = tf.sequential();

  model.add(tf.layers.dense({
    inputShape: [1],
    units: 200,
    activation: 'relu'
  }));

  model.add(tf.layers.dense({
    units: 200,
    activation: 'relu'
  }));

  model.add(tf.layers.dense({
    units: 200,
    activation: 'relu'
  }));

  model.add(tf.layers.dense({
    units: 1
  }));

  return model;
}

function convertToTensor(data) {
  return tf.tidy(() => {
    const inputs = data.map(point => [point.x]);
    const labels = data.map(point => [point.y]);

    const inputTensor = tf.tensor2d(inputs);
    const labelTensor = tf.tensor2d(labels);

    return {
      inputs: inputTensor,
      labels: labelTensor
    };
  });
}

async function trainModel(model, trainData, epochs, trainingName) {
  const tensorData = convertToTensor(trainData);
  const inputs = tensorData.inputs;
  const labels = tensorData.labels;

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'meanSquaredError',
    metrics: ['mse']
  });

  console.log(`${trainingName} gestartet...`);

  const history = await model.fit(inputs, labels, {
    batchSize: 32,
    epochs: epochs,
    shuffle: true
  });

  console.log(`${trainingName} beendet.`);

  inputs.dispose();
  labels.dispose();

  return history;
}

function calculateMSE(model, data) {
  return tf.tidy(() => {
    const tensorData = convertToTensor(data);
    const inputs = tensorData.inputs;
    const labels = tensorData.labels;

    const predictions = model.predict(inputs);

    const mseTensor = predictions
      .sub(labels)
      .square()
      .mean();

    return mseTensor.dataSync()[0];
  });
}

function createPredictionCurve(model, min, max, numberOfPoints) {
  return tf.tidy(() => {
    const xs = tf.linspace(min, max, numberOfPoints);
    const inputs = xs.reshape([numberOfPoints, 1]);

    const predictions = model.predict(inputs);

    const xValues = xs.arraySync();
    const yValues = predictions.arraySync();

    return xValues.map((x, index) => {
      return {
        x: x,
        y: yValues[index][0]
      };
    });
  });
}