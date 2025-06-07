# Funktionsweise des Systems

## Entscheidungslogik

Der Agent wählt in jedem Zustand eine Aktion basierend auf der Q-Tabelle. Dabei wird eine Epsilon-Greedy-Strategie verwendet: Mit Wahrscheinlichkeit `epsilon` wird eine zufällige Aktion gewählt (Exploration), sonst die beste bekannte Aktion (Exploitation).

## Zustandsrepräsentation

- **Grid-Umgebungen**: Einzelner Integer-Wert (0-24 für 5x5 Gitter)
- **Container-Umgebung**: Erweitert um Ladezustand - codiert als `x * 5 + y + 25 * container_loaded`

## Belohnungsstruktur

| Ereignis | Belohnung |
|----------|-----------|
| Ziel erreicht | +10 |
| Container aufgenommen | +8 |
| Container abgeliefert | +20 |
| Normaler Schritt | -1 |
| Hinderniskollision | -10 |
| Schleifenabbruch | -10 |
| Timeout | -10 |

## Terminierungsbedingungen

Das System erkennt verschiedene Episode-Enden in dieser Prioritätsreihenfolge:
1. **Erfolg**: Ziel erreicht oder Container erfolgreich abgeliefert
2. **Schleifenerkennung**: Zustand wird öfter als `LOOP_THRESHOLD` besucht
3. **Hinderniskollision**: Agent trifft auf Hindernis
4. **Timeout**: Maximale Schrittanzahl (`MAX_STEPS`) erreicht

## Modussteuerung

Über `config.py` werden verschiedene Szenarien gesteuert:
- **`static`**: Feste Positionen für Start, Ziel und Hindernisse
- **`random_start`**: Zufällige Startposition
- **`random_goal`**: Zufällige Zielposition  
- **`random_obstacles`**: Zufällige Hindernispositionen
- **`container`**: Pickup/Dropoff-Aufgabe mit erweitertem Zustandsraum

## Q-Tabellen-Verwaltung

Jedes Szenario erhält eine eigene Q-Tabelle:
- `q_table_static.npy`
- `q_table_random_start.npy`
- `q_table_container.npy`
- usw.

Die Q-Tabellen werden automatisch geladen und gespeichert basierend auf dem gewählten `ENV_MODE`.