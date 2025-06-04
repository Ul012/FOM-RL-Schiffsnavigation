# Willkommen

Mit diesem Projekt navigiert ein Reinforcement-Learning-Agent ein Schiff durch eine Gitterwelt.

## Ziele
- Entwicklung eines Q-Learning-Agents
- Erstellung einer eigenen OpenAI-Gymnasium-Umgebung
- Dokumentation des Projektverlaufs mit MkDocs

👉 Siehe [Setup & Ausführung](setup.md)

---

## 🔀 Modus-Übersicht

| Modus         | Beschreibung                                                                 |
|---------------|------------------------------------------------------------------------------|
| `static`      | Fester Start-, Ziel- und Hindernisbereich. Gut zum Einstieg, stabile Lernkurve. |
| `random_start`| Zufälliger Startpunkt bei festem Ziel. Testet Robustheit beim Startverhalten. |
| `random_goal` | Fester Startpunkt, aber zufälliges Ziel. Erfordert flexible Zielnavigation.   |
| `random_obstacles` | Hindernisse variieren pro Episode. Erhöht Unsicherheit, erschwert Lernen. |
| `container`   | Zwei-Ziel-Aufgabe: Container muss zuerst abgeholt, dann zum Ziel gebracht werden. |

---

## 📁 Projektstruktur

```text
FOM-rl-shipnav-qlearning/
├── src/
│   ├── train.py                     ← Q-Learning-Training mit Visualisierung & Erfolgsmetrik
│   ├── evaluate_policy.py           ← Statistische Zielerreichung auf zufälligen Karten
│   ├── visualize_policy.py          ← Statische Darstellung der gelernten Policy mit Emojis & Exportfunktion
│   ├── config.py                    ← Zentrale Steuerung des Szenarios über ENV_MODE
│   ├── q_table.py                   ← Laden/Speichern von Q-Tabellen
│   └── navigation/
│       └── environment/
│           ├── grid_environment.py      ← Basisumgebung
│           └── container_environment.py ← Container-Szenario mit Pickup/Dropoff
├── requirements.txt                 ← Projektabhängigkeiten
├── mkdocs.yml                       ← MkDocs-Konfiguration
├── docs/
│   ├── index.md                     ← Startseite
│   ├── setup.md                     ← Setup-Anleitung für Umgebung, Training und Visualisierung
└── site/                            ← (von mkdocs build erzeugt)
```

---

## Ablauf des Projekts

Der Ablauf des Projekts gliedert sich in drei Hauptschritte:

1. **Training**  
   Das Skript `train.py` erstellt eine Q-Tabelle durch Interaktion des Agenten mit der Umgebung. Dabei wird der Lernfortschritt über Rewards und Zielerreichung pro Episode dokumentiert. Über die Variable `ENV_MODE` in `config.py` kann zwischen verschiedenen Szenarien wie `static`, `random_start`, `random_goal` oder `container` gewählt werden.

2. **Evaluation**  
   Mit `evaluate_policy.py` wird die gelernte Policy getestet – z. B. in 100 zufällig generierten Umgebungen. Es erfolgt kein Lernen mehr: Der Agent folgt der gespeicherten Q-Tabelle (`q_table.npy`) und wählt jeweils die beste bekannte Aktion. Ziel ist es, Erfolgsquote und durchschnittlichen Reward zu ermitteln. Es wird auch geprüft, ob sich der Agent in einer Endlosschleife befindet.

3. **Visualisierung**  
   `visualize_policy.py` zeigt einen einzelnen Lauf des Agenten in der Umgebung animiert mit Pygame. Diese Darstellung dient der qualitativen Demonstration des Lernverhaltens. Optional können ein GIF und ein PDF-Screenshot exportiert werden.