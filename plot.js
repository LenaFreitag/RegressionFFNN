function plotDataset(containerId, trainData, testData, title) {
  const trainTrace = {
    x: trainData.map(point => point.x),
    y: trainData.map(point => point.y),
    mode: 'markers',
    type: 'scatter',
    name: 'Training'
  };

  const testTrace = {
    x: testData.map(point => point.x),
    y: testData.map(point => point.y),
    mode: 'markers',
    type: 'scatter',
    name: 'Test'
  };

  const layout = {
    title: title,
    xaxis: {
      title: 'x'
    },
    yaxis: {
      title: 'y'
    }
  };

  Plotly.newPlot(containerId, [trainTrace, testTrace], layout);
}

function plotPrediction(containerId, data, predictionCurve, title) {
  const dataTrace = {
    x: data.map(point => point.x),
    y: data.map(point => point.y),
    mode: 'markers',
    type: 'scatter',
    name: 'Daten'
  };

  const predictionTrace = {
    x: predictionCurve.map(point => point.x),
    y: predictionCurve.map(point => point.y),
    mode: 'lines',
    type: 'scatter',
    name: 'Modellvorhersage'
  };

  const layout = {
    title: title,
    xaxis: {
      title: 'x'
    },
    yaxis: {
      title: 'y'
    }
  };

  Plotly.newPlot(containerId, [dataTrace, predictionTrace], layout);
}

function showMSE(elementId, label, mse) {
  const element = document.getElementById(elementId);

  if (!element) {
    console.warn(`MSE-Element mit ID "${elementId}" wurde nicht gefunden.`);
    return;
  }

  element.textContent = `${label}: MSE = ${mse.toFixed(5)}`;
}