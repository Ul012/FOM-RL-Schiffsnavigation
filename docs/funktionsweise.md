# Funktionsweise des Agenten

Dieses Kapitel beschreibt die grundlegende Funktionsweise des Q-Learning-Agenten.

## Entscheidungslogik

Der Agent wählt in jedem Zustand eine Aktion, die basierend auf der Q-Tabelle die höchste Belohnung verspricht. Die Q-Tabelle wird dabei iterativ verbessert.

## Belohnungsstruktur

- Ziel erreicht: +10 Punkte
- Hindernis getroffen oder Schleife erkannt: -10 Punkte
- Jeder Schritt: -0.1 Punkte

## Modussteuerung

Über `config.py` kann der Modus gesteuert werden:
- `static`, `random_start`, `random_goal`, `random_obstacles`, `container`
