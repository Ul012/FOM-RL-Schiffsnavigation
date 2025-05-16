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
│   ├── train.py                     ← Q-Learning-Training mit Visualisierung & Erfolgsmetrik
│   ├── evaluate_policy.py           ← Statistische Zielerreichung auf zufälligen Karten
│   ├── run_policy.py                ← Animierter Agentenlauf (optional)
│   ├── visualize_policy.py          ← Statische Darstellung der gelernten Policy mit Emojis
│   └── navigation/
│       └── environment/
│           └── grid_environment.py  ← Gym-kompatible Umgebung mit Moduswahl & Seed
├── requirements.txt                 ← Projektabhängigkeiten
├── mkdocs.yml                       ← MkDocs-Konfiguration
├── docs/
│   ├── index.md                     ← Startseite
│   ├── funktionsweise.md            ← Q-Learning, Zustände, Aktionen
│   ├── training.md                  ← Lernkurven, Metriken, Modusvergleich
│   └── how-to-push.md               ← Anleitung zum Git-Push mit PyCharm oder Terminal
└── site/                            ← (von mkdocs build erzeugt)
```
