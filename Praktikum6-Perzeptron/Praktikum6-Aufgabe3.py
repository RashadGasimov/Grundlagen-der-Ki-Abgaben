# perceptron_experiment_extended.py
# Erweiterte Version des Perzeptron-Experiments
# Mit klar getrennten Abschnitten: Datensatz, Training und Experimente

import random
import numpy as np


# ================================================================
# ======================== 1. DATENSATZ ==========================
# ================================================================

def generate_dataset(m, rng):
    """
    Erzeugt einen Datensatz mit:
    - m Punkten im Bereich [-1, 1] × [-1, 1]
    - zufälliger Zielgerade (Hypothese f)
    - Ziel-Labels (+1 / -1)

    Rückgabe:
        X = Array der Eingabepunkte
        y = Array der zugehörigen Labels
    """

    # m zufällige Punkte im 2D-Bereich
    X = rng.uniform(-1, 1, (m, 2))

    # Zwei zufällige Punkte definieren die Zielgerade
    p1 = rng.uniform(-1, 1, 2)
    p2 = rng.uniform(-1, 1, 2)

    # Normalenvektor der Geraden
    w_true = np.array([
        p2[1] - p1[1],
        -(p2[0] - p1[0])
    ])

    # Bias der Geraden
    b_true = - w_true.dot(p1)

    # Label-Funktion
    def label(x):
        return 1 if w_true.dot(x) + b_true >= 0 else -1

    # Alle Labels erzeugen
    y = np.array([label(x) for x in X])

    return X, y



# ================================================================
# ======================== 2. TRAINING ===========================
# ================================================================

def perceptron_run(X, y, alpha=1.0, max_steps=200000):
    """
    Führt den Perzeptron-Lernalgorithmus aus.

    Eingaben:
        X = Datenpunkte
        y = Labels (+1/-1)
        alpha = Lernrate
        max_steps = Sicherheitslimit

    Rückgabe:
        Anzahl durchgeführter Updates (Schritte)
    """

    m = len(X)

    # Startgewichte: w = (0,0), Bias = 0
    w = np.zeros(2)
    b = 0
    steps = 0

    # Training läuft, bis alle Punkte korrekt klassifiziert sind
    while True:
        misclassified = []

        # Falsch klassifizierte Punkte bestimmen
        for i in range(m):
            prediction = 1 if (w.dot(X[i]) + b) >= 0 else -1
            if prediction != y[i]:
                misclassified.append(i)

        # Wenn nichts falsch ist → konvergiert
        if not misclassified:
            break

        # Zufälligen Fehlerpunkt auswählen (wie in der Aufgabe gefordert)
        i = random.choice(misclassified)

        # Perzeptron-Update:
        # w := w + alpha * y_i * x_i
        # b := b + alpha * y_i
        w = w + alpha * y[i] * X[i]
        b = b + alpha * y[i]

        steps += 1

        # Sicherheitsgrenze (falls etwas schiefgeht)
        if steps > max_steps:
            break

    return steps



# ================================================================
# ======================= 3. EXPERIMENT ==========================
# ================================================================

def experiment(m, alpha, runs=100, seed=42):
    """
    Führt das vollständige Experiment mehrmals aus:

    - Erzeugt jeweils einen neuen Datensatz der Größe m
    - Führt das Perzeptron-Training mit Lernrate alpha durch
    - Wiederholt das Ganze 'runs' Mal
    - Berechnet Mittelwert und Standardabweichung der Updates

    Rückgabe:
        (durchschnittliche Schritte, Standardabweichung)
    """

    rng = np.random.RandomState(seed)
    random.seed(seed)

    results = []

    for _ in range(runs):
        X, y = generate_dataset(m, rng)
        steps = perceptron_run(X, y, alpha=alpha)
        results.append(steps)

    return np.mean(results), np.std(results)



# ================================================================
# ============== 4. HAUPTPROGRAMM – AUSGABE DER EXPERIMENTE ======
# ================================================================

if __name__ == "__main__":

    # Die vier geforderten Kombinationen aus der Aufgabe
    settings = [
        (100, 1.0),
        (100, 0.1),
        (1000, 1.0),
        (1000, 0.1)
    ]

    # Ergebnisse ausgeben
    for m, alpha in settings:
        avg, std = experiment(m, alpha)
        print(f"m={m}, alpha={alpha}: Durchschnittliche Updates = {avg:.2f} ± {std:.2f}")
