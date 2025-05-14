# Simulation von Schiffsnavigation mit Reinforcement Learning

Dieses Projekt implementiert einen agentenbasierten Ansatz zur optimalen Routenfindung in einer Gitterwelt mithilfe von Q-Learning.

## Inhalte
- Eigene Grid-Umgebung (OpenAI Gym-kompatibel)
- Q-Learning-Implementierung
- Visualisierung von Lernverläufen
- Dokumentation mit MkDocs

## Setup
```bash
python -m venv rl-venv
rl-venv\Scripts\activate
pip install -r requirements.txt
cd src
python train.py
