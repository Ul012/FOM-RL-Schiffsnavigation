# Training des Agenten

Das Training erfolgt über `train.py`. In jeder Episode wird eine neue Umgebung generiert (abhängig vom Modus), und der Agent lernt durch Interaktion.

## Ablauf einer Episode

1. Umgebung wird initialisiert
2. Agent bewegt sich bis zum Ziel oder Timeout
3. Q-Werte werden basierend auf Reward aktualisiert

## Hyperparameter

- Lernrate (`alpha`)
- Diskontfaktor (`gamma`)
- Explorationsrate (`epsilon`) mit optionalem Decay
