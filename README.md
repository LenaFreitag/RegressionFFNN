# Regression mit TensorFlow.js

Dieses Projekt zeigt eine Regression mit einem Feed-Forward Neural Network in TensorFlow.js. Ziel ist es, eine unbekannte Funktion anhand erzeugter Daten zu approximieren und den Unterschied zwischen gutem Fit und Overfitting sichtbar zu machen.

## Inhalt

Die Anwendung erzeugt zufällige Datenpunkte im Intervall `[-2, 2]`. Daraus werden unverrauschte und verrauschte Trainings- und Testdaten erstellt. Anschließend werden drei Modelle verglichen:

* **Clean Model:** trainiert auf unverrauschten Daten
* **Best-Fit Model:** trainiert auf verrauschten Daten mit guter Generalisierung
* **Overfit Model:** trainiert auf verrauschten Daten mit zu vielen Epochen

## Technologien

* HTML
* CSS
* JavaScript
* TensorFlow.js
* Chart.js / Plotly

## Modell

Verwendet wird ein Feed-Forward Neural Network mit:

* 2 Hidden Layern mit je 100 Neuronen
* ReLU-Aktivierung
* linearem Output
* Mean Squared Error als Loss
* Adam Optimizer

## Start

```bash
npx serve .
```

Dann die Anwendung im Browser öffnen.

## Ziel

Die Visualisierungen zeigen, wie gut die Modelle die Daten approximieren. Besonders der Vergleich zwischen Best-Fit und Overfit macht deutlich, dass ein niedriger Trainings-Loss nicht automatisch eine gute Generalisierung bedeutet.
