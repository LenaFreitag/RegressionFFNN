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
* Plotly.js

## Modell

Verwendet wird ein Feed-Forward Neural Network mit:

* 2 Hidden Layern mit je 100 Neuronen
* ReLU-Aktivierung
* linearem Output
* Mean Squared Error als Loss
* Adam Optimizer

**Model-Ordnerstruktur**

Die vortrainierten Modelle liegen in Unterordnern unter `models/`:

```
models/
├── clean-model/
│   ├── clean-model.json
│   └── clean-model.weights.bin
├── best-fit-model/
│   ├── bestfit-model.json
│   └── bestfit-model.weights.bin
└── overfit-model/
	├── overfit-model.json
	└── overfit-model.weights.bin
```

**Wichtige Hyperparameter (used in this project)**

* Datensätze: `N = 100` (Aufteilung `50` Train / `50` Test)
* Rauschen: Gaussian Noise, `variance = 0.05` (nur y-Labels)
* Optimizer: `Adam` mit `learningRate = 0.01`
* `batchSize = 32`
* Epochs: `clean = 500`, `bestFit = 1000`, `overfit = 3000`

## Start

```bash
npx serve .
```

Die Anwendung im Browser öffnen. Hinweis: Die Seite lädt standardmäßig die in `models/` gespeicherten, vortrainierten Modelle und berechnet Vorhersagen und MSE beim Laden. Für eigene Experimente sind Funktionen zum Erzeugen, Trainieren und Speichern von Modellen im Quellcode enthalten.

## Ziel

Die Visualisierungen zeigen, wie gut die Modelle die Daten approximieren. Besonders der Vergleich zwischen Best-Fit und Overfit macht deutlich, dass ein niedriger Trainings-Loss nicht automatisch eine gute Generalisierung bedeutet.
