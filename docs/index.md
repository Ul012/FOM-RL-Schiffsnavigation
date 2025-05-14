# Willkommen

Dieses Projekt demonstriert, wie ein Reinforcement-Learning-Agent ein Schiff durch eine Gitterwelt navigiert.

## Ziele
- Entwicklung eines Q-Learning-Agents
- Erstellung einer eigenen OpenAI-Gymnasium-Umgebung
- Dokumentation des Projektverlaufs mit MkDocs

👉 Siehe [Setup & Ausführung](setup.md)

## Projektstruktur

## 📁 Projektstruktur

```text
FOM-rl-shipnav-qlearning/
├── src/
│   ├── train.py                     ← Q-Learning Training
│   ├── run_policy.py                ← Agentenlauf nach gelernter Policy
│   ├── visualize_policy.py          ← Statische Visualisierung
│   └── navigation/
│       └── environment/
│           └── grid_environment.py  ← Gym-kompatible Umgebung mit Moduswahl
├── requirements.txt                 ← Projektabhängigkeiten
├── mkdocs.yml                       ← MkDocs-Konfiguration
├── docs/
│   ├── index.md                     ← Startseite
│   ├── funktionsweise.md            ← Q-Learning & Agent
│   ├── training.md                  ← Lernkurve und Ergebnisse
│   └── how-to-push.md               ← Git-Push-Anleitung
└── site/                            ← (von mkdocs build erzeugt)
```
