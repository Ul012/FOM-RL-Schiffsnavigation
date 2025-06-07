# Visualisierung und Analyse

## Policy-Darstellung

```bash
python visualize_policy.py
```

Zeigt die gelernte Policy als animierte Darstellung mit Symbolen:
- 🧭 Startposition
- 🚢 Aktueller Agent  
- 🏁 Zielposition
- 🪨 Hindernis
- 📤 Pickup-Punkt (Container-Modus)
- 📦 Dropoff-Punkt (Container-Modus)

## Szenarien-Vergleich

```bash
python compare_scenarios.py
```

Erstellt automatisch drei Visualisierungen:

### 1. Erfolgsraten-Vergleich
Balkendiagramm der Erfolgsraten aller Szenarien

### 2. Terminierungsarten-Analyse  
Gestapeltes Balkendiagramm zeigt die Verteilung von:
- Erfolg (grün)
- Timeout (rot)
- Schleifenabbruch (orange) 
- Hindernis-Kollision (braun)

### 3. Statistische Tabelle
Tabellarische Übersicht mit detaillierten Metriken pro Szenario:
- Erfolgsraten in Prozent
- Verteilung der verschiedenen Terminierungsarten
- Durchschnittliche Rewards und Schrittanzahlen
- Statistische Kennwerte (Standardabweichungen)

## Q-Tabellen-Inspektion

```bash
python inspect_q_tables.py
```

Interaktive Analyse mit Optionen:
1. Aktuelles Szenario (aus config.py)
2. Spezifisches Szenario wählen
3. Alle verfügbaren Q-Tabellen
4. Formen-Vergleich
5. Nur Matrix anzeigen

Zeigt Q-Werte, Statistiken und komplette Matrix-Darstellung.

## Export-Funktionen

Alle Visualisierungen werden automatisch als PDF exportiert:
- `exports/train_learning_curve.pdf`
- `exports/success_rates.pdf` 
- `exports/failure_modes.pdf`

Konfiguration über:
```python
EXPORT_PDF = True
EXPORT_PATH = "exports/"
```

## Interpretation

### Erfolgreiche Policies
- Hohe Erfolgsrate (>80%)
- Niedrige Timeout-Rate (<10%)
- Stabile Lernkurven

### Problematische Policies  
- Niedrige Erfolgsrate (<50%)
- Hohe Timeout- oder Schleifenraten (>30%)
- Instabile Lernkurven

Das System ermöglicht schnelle visuelle Bewertung der Trainingsqualität und Vergleich verschiedener Szenarien.