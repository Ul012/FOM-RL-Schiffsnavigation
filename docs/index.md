# 🚢 Q-Learning für Schiffsnavigation
Dieses Projekt implementiert einen Q-Learning-Algorithmus zur autonomen Navigation eines Agenten durch verschiedene Gitterumgebungen. Das System ermöglicht das Training und die Evaluation von Navigationsstrategien unter verschiedenen Umgebungsbedingungen.

## 🎯 Projektziele

- Entwicklung eines robusten Q-Learning-Agenten für Navigationsprobleme
- Implementierung verschiedener Umgebungsszenarien mit unterschiedlichen Komplexitätsgraden
- Bereitstellung von Evaluations- und Vergleichstools für wissenschaftliche Analyse
- Dokumentation der Implementierung und Ergebnisse

## 🏗️ Systemarchitektur

Das Projekt besteht aus mehreren Komponenten:

- **Training**: Automatisiertes Lernen für einzelne oder multiple Szenarien
- **Evaluation**: Quantitative Analyse der gelernten Policies
- **Vergleich**: Statistische Auswertung verschiedener Szenarien
- **Visualisierung**: Grafische Darstellung der Agentenverhalten und Lernfortschritte

## 🗺️ Verfügbare Umgebungsszenarien

| Szenario | Beschreibung | Anwendungsbereich |
|----------|--------------|-------------------|
| **Static** | Konstante Positionen für alle Elemente | Grundlegendes Q-Learning |
| **Random Start** | Variable Startpositionen | Robustheitstesting |
| **Random Goal** | Variable Zielpositionen | Adaptive Navigation |
| **Random Obstacles** | Variable Hindernisverteilungen | Dynamische Umgebungen |
| **Container** | Pickup/Dropoff-Aufgaben | Komplexe Aufgabenstellungen |

## ⚙️ Technische Spezifikationen

- **Umgebung**: 5x5 Gitterwelt (OpenAI Gymnasium-kompatibel)
- **Algorithmus**: Q-Learning mit Epsilon-Greedy-Exploration
- **Zustandsraum**: Diskret (25 Zustände für Grid, erweitert für Container)
- **Aktionsraum**: 4 Bewegungsrichtungen (Oben, Rechts, Unten, Links)
- **Terminierungsbedingungen**: Zielerreichung, Timeout, Schleifenerkennung, Hinderniskollision

## 📊 Wissenschaftliche Evaluierung

Das System bietet umfassende Analysemöglichkeiten:

- Erfolgsraten-Vergleich zwischen verschiedenen Szenarien
- Statistische Auswertung von Terminierungsarten
- Lernkurven-Analyse mit Moving-Average-Glättung
- Parameter-Sensitivitätsanalyse
- Reproduzierbare Experimente durch zentrale Konfiguration

---

**📚 Weiterführende Informationen:**

- [Setup und Installation](setup.md)
- [Funktionsweise des Systems](funktionsweise.md)
- [Training und Konfiguration](training.md)
- [Visualisierung und Export](visualisierung.md)