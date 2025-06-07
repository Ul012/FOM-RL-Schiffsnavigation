# train_all_scenarios.py

# ============================================================================
# Imports
# ============================================================================

import sys
import os
import time
from pathlib import Path
import subprocess
from datetime import datetime

# Projektstruktur für Import anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Drittanbieter
import numpy as np
import matplotlib.pyplot as plt

# Lokale Module - direkte Imports für Training
from navigation.environment.grid_environment import GridEnvironment
from navigation.environment.container_environment import ContainerShipEnv

# ============================================================================
# Konfiguration
# ============================================================================

# Alle zu trainierenden Szenarien
SCENARIOS = {
    "static": {
        "env_mode": "static",
        "environment": "grid",
        "description": "Statisches Grid (feste Start-, Ziel- und Hindernis-Positionen)"
    },
    "random_start": {
        "env_mode": "random_start",
        "environment": "grid",
        "description": "Zufällige Startposition"
    },
    "random_goal": {
        "env_mode": "random_goal",
        "environment": "grid",
        "description": "Zufällige Zielposition"
    },
    "random_obstacles": {
        "env_mode": "random_obstacles",
        "environment": "grid",
        "description": "Zufällige Hindernis-Positionen"
    },
    "container": {
        "env_mode": "container",
        "environment": "container",
        "description": "Container-Schiff Umgebung mit Pickup/Dropoff"
    }
}

# Visualisierung und Training-Modi
SHOW_VISUALIZATIONS = True
PARALLEL_TRAINING = False


# ============================================================================
# Hilfsfunktionen
# ============================================================================

def load_current_config():
    """Aktuelle config.py Parameter laden"""
    # config.py als Modul importieren
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", "config.py")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # Parameter extrahieren
    config_params = {
        "EPISODES": config_module.EPISODES,
        "MAX_STEPS": config_module.MAX_STEPS,
        "ALPHA": config_module.ALPHA,
        "GAMMA": config_module.GAMMA,
        "EPSILON": config_module.EPSILON,
        "LOOP_THRESHOLD": config_module.LOOP_THRESHOLD,
        "EXPORT_PDF": config_module.EXPORT_PDF,
        "EXPORT_PATH": config_module.EXPORT_PATH
    }

    print("✅ Parameter aus config.py geladen:")
    for key, value in config_params.items():
        print(f"  {key}: {value}")

    return config_params


def setup_training_environment(config_params):
    """Training-Umgebung vorbereiten"""
    # Export-Ordner erstellen
    Path(config_params["EXPORT_PATH"]).mkdir(exist_ok=True)

    # Matplotlib Backend konfigurieren
    if not SHOW_VISUALIZATIONS:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        print("📊 Matplotlib: Non-interactive Backend (keine Visualisierungen)")
    else:
        print("📊 Matplotlib: Interactive Backend (Visualisierungen werden angezeigt)")

    print(f"Training-Setup abgeschlossen:")
    print(f"  Episodes: {config_params['EPISODES']}")
    print(f"  Max Steps: {config_params['MAX_STEPS']}")
    print(f"  Loop Threshold: {config_params['LOOP_THRESHOLD']}")
    print(f"  Export-Pfad: {config_params['EXPORT_PATH']}")
    print(f"  Visualisierungen: {'Ja (interaktiv)' if SHOW_VISUALIZATIONS else 'Nein (nur PDF-Export)'}")


def update_config_for_scenario(scenario_name, scenario_config, config_params):
    """config.py für Szenario aktualisieren"""
    config_content = f'''# config.py - Auto-generiert für Szenario: {scenario_name}
# Generiert von train_all_scenarios.py am {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# ============================================================================
# Umgebungsparameter
# ============================================================================

ENV_MODE = "{scenario_config["env_mode"]}"
GRID_SIZE = 5

# ============================================================================
# Training-Parameter
# ============================================================================

EPISODES = {config_params["EPISODES"]}
MAX_STEPS = {config_params["MAX_STEPS"]}
ALPHA = {config_params["ALPHA"]}
GAMMA = {config_params["GAMMA"]}
EPSILON = {config_params["EPSILON"]}
LOOP_THRESHOLD = {config_params["LOOP_THRESHOLD"]}

# ============================================================================
# Aktionen und Rewards (aus ursprünglicher config.py)
# ============================================================================

ACTIONS = {{
    "up": 0,
    "right": 1, 
    "down": 2,
    "left": 3
}}

REWARDS = {{
    "goal": 10,
    "step": -1,
    "obstacle": -10,
    "loop_abort": -15,
    "timeout": -20,
    "pickup": 5,
    "dropoff": 15
}}

# ============================================================================
# Pfade und Export
# ============================================================================

Q_TABLE_PATH = "q_table_{scenario_config["env_mode"]}.npy"
EXPORT_PDF = {str(config_params["EXPORT_PDF"]) if SHOW_VISUALIZATIONS else "False"}
EXPORT_PATH = "{config_params["EXPORT_PATH"]}"

# ============================================================================
# Matplotlib Backend (für automatisiertes Training)
# ============================================================================

import matplotlib
{"matplotlib.use('Agg')  # Non-interactive backend" if not SHOW_VISUALIZATIONS else "# Interactive backend aktiv"}
'''

    with open("config.py", "w", encoding="utf-8") as f:
        f.write(config_content)


def backup_config():
    """config.py sichern"""
    if os.path.exists("config.py"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"config_backup_{timestamp}.py"

        with open("config.py", "r", encoding="utf-8") as original:
            content = original.read()

        with open(backup_path, "w", encoding="utf-8") as backup:
            backup.write(content)

        print(f"Config-Backup erstellt: {backup_path}")


def run_training_for_scenario(scenario_name, scenario_config, config_params):
    """Training für einzelnes Szenario ausführen"""
    print(f"\n{'=' * 60}")
    print(f"STARTE TRAINING: {scenario_name.upper()}")
    print(f"{'=' * 60}")
    print(f"Beschreibung: {scenario_config['description']}")
    print(f"Modus: {scenario_config['env_mode']}")
    print(f"Umgebung: {scenario_config['environment']}")
    print(f"Episodes: {config_params['EPISODES']}")

    start_time = time.time()

    try:
        # Backup der ursprünglichen config.py
        original_config = None
        if os.path.exists("config.py"):
            with open("config.py", "r", encoding="utf-8") as f:
                original_config = f.read()

        # Config für dieses Szenario aktualisieren
        update_config_for_scenario(scenario_name, scenario_config, config_params)

        # Training starten
        result = subprocess.run([sys.executable, "train.py"],
                                capture_output=True, text=True, timeout=1800)  # 30min timeout

        # Ursprüngliche config.py wiederherstellen
        if original_config:
            with open("config.py", "w", encoding="utf-8") as f:
                f.write(original_config)

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ Training erfolgreich abgeschlossen ({duration:.1f}s)")
            print(f"Q-Tabelle gespeichert: q_table_{scenario_config['env_mode']}.npy")

            # Letzten Teil der Ausgabe zeigen
            output_lines = result.stdout.split('\n')[-10:]
            for line in output_lines:
                if line.strip():
                    print(f"  {line}")

        else:
            print(f"❌ Training fehlgeschlagen ({duration:.1f}s)")
            print(f"Fehler: {result.stderr}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        # Config wiederherstellen auch bei Timeout
        if original_config:
            with open("config.py", "w", encoding="utf-8") as f:
                f.write(original_config)
        print(f"⏰ Training-Timeout nach 30min")
        return False
    except Exception as e:
        # Config wiederherstellen auch bei Fehler
        if original_config:
            with open("config.py", "w", encoding="utf-8") as f:
                f.write(original_config)
        print(f"❌ Unerwarteter Fehler: {e}")
        return False


def run_parallel_training():
    """Alle Szenarien parallel trainieren"""
    print("🚀 PARALLEL TRAINING GESTARTET")
    print("Achtung: Parallel Training kann zu Ressourcen-Konflikten führen!")

    processes = []

    for scenario_name, scenario_config in SCENARIOS.items():
        # Separate Python-Instanz für jedes Szenario
        cmd = [sys.executable, "-c", f"""
import sys
sys.path.append('.')
from train_all_scenarios import run_training_for_scenario
scenario_config = {scenario_config}
run_training_for_scenario('{scenario_name}', scenario_config)
"""]

        process = subprocess.Popen(cmd)
        processes.append((scenario_name, process))
        print(f"Gestartet: {scenario_name} (PID: {process.pid})")

    # Auf alle Prozesse warten
    results = {}
    for scenario_name, process in processes:
        process.wait()
        results[scenario_name] = process.returncode == 0
        print(f"Beendet: {scenario_name} ({'✅' if results[scenario_name] else '❌'})")

    return results


def run_sequential_training(config_params):
    """Alle Szenarien nacheinander trainieren"""
    print("🔄 SEQUENZIELLES TRAINING GESTARTET")

    results = {}
    total_start = time.time()

    for i, (scenario_name, scenario_config) in enumerate(SCENARIOS.items(), 1):
        print(f"\n[{i}/{len(SCENARIOS)}] Nächstes Szenario: {scenario_name}")
        results[scenario_name] = run_training_for_scenario(scenario_name, scenario_config, config_params)

        # Kurze Pause zwischen Trainings
        time.sleep(2)

    total_duration = time.time() - total_start
    print(f"\n🏁 GESAMTES TRAINING ABGESCHLOSSEN ({total_duration / 60:.1f} min)")

    return results


def create_training_summary(results):
    """Zusammenfassung des Trainings erstellen"""
    print(f"\n{'=' * 60}")
    print("TRAINING ZUSAMMENFASSUNG")
    print(f"{'=' * 60}")

    successful = 0
    failed = 0

    for scenario_name, success in results.items():
        status = "✅ Erfolgreich" if success else "❌ Fehlgeschlagen"
        q_table_exists = os.path.exists(f"q_table_{SCENARIOS[scenario_name]['env_mode']}.npy")
        q_table_status = "Q-Tabelle ✓" if q_table_exists else "Q-Tabelle ✗"

        print(f"{scenario_name:<20} {status:<15} {q_table_status}")

        if success:
            successful += 1
        else:
            failed += 1

    print(f"\nErgebnis: {successful}/{len(SCENARIOS)} erfolgreich")

    if successful == len(SCENARIOS):
        print("🎉 Alle Szenarien erfolgreich trainiert!")
        print("Sie können jetzt 'python compare_scenarios.py' ausführen.")
    elif successful > 0:
        print(f"⚠️  {failed} Szenarien fehlgeschlagen - Vergleich mit verfügbaren Daten möglich.")
    else:
        print("💥 Alle Trainings fehlgeschlagen - Bitte Konfiguration prüfen.")


def restore_original_config():
    """Diese Funktion ist nicht mehr nötig"""
    pass  # Entfernt - config.py wird nach jedem Training direkt wiederhergestellt


# ============================================================================
# Hauptfunktion
# ============================================================================

def train_all_scenarios():
    """Hauptfunktion für Training aller Szenarien"""
    print("🏗️  MULTI-SZENARIO TRAINING")
    print(f"Anzahl Szenarien: {len(SCENARIOS)}")
    print(f"Training-Modus: {'Parallel' if PARALLEL_TRAINING else 'Sequenziell'}")

    # Aktuelle config.py Parameter laden
    config_params = load_current_config()

    # Setup
    setup_training_environment(config_params)

    # Training ausführen
    if PARALLEL_TRAINING:
        results = run_parallel_training()
    else:
        results = run_sequential_training(config_params)

    # Zusammenfassung
    create_training_summary(results)

    return results


# ============================================================================
# Interaktive Funktionen
# ============================================================================

def select_scenarios_interactively():
    """Interaktive Szenario-Auswahl"""
    print("\nVerfügbare Szenarien:")
    scenario_list = list(SCENARIOS.items())

    for i, (name, config) in enumerate(scenario_list, 1):
        print(f"{i}. {name}: {config['description']}")

    print(f"{len(scenario_list) + 1}. Alle Szenarien")

    try:
        choice = input(f"\nWählen Sie Szenarien (1-{len(scenario_list) + 1}, kommagetrennt): ").strip()

        if choice == str(len(scenario_list) + 1):
            return SCENARIOS

        selected_indices = [int(x.strip()) - 1 for x in choice.split(',')]
        selected_scenarios = {}

        for idx in selected_indices:
            if 0 <= idx < len(scenario_list):
                name, config = scenario_list[idx]
                selected_scenarios[name] = config

        return selected_scenarios

    except (ValueError, IndexError):
        print("Ungültige Eingabe - alle Szenarien werden trainiert")
        return SCENARIOS


def configure_training_interactively():
    """Interaktive Training-Konfiguration"""
    global SHOW_VISUALIZATIONS

    # Aktuelle config.py Parameter laden und anzeigen
    config_params = load_current_config()

    # Visualisierungs-Modus abfragen
    print(f"\n📊 VISUALISIERUNGS-MODUS:")
    print("1. Interaktive Visualisierungen (Sie müssen jedes Diagramm schließen)")
    print("2. Automatisiert (nur PDF-Export, keine interaktiven Fenster)")

    viz_choice = input("Wählen Sie Modus (1/2): ").strip()
    SHOW_VISUALIZATIONS = (viz_choice != '2')

    if SHOW_VISUALIZATIONS:
        print("✅ Interaktive Visualisierungen aktiviert")
        print("⚠️  Sie müssen jedes Matplotlib-Fenster manuell schließen!")
    else:
        print("✅ Automatisierter Modus aktiviert")
        print("📄 Alle Diagramme werden nur als PDF gespeichert")

    print(f"\n💡 Alle Training-Parameter kommen aus der config.py")
    print(f"💡 Zum Ändern der Parameter bearbeiten Sie die config.py direkt")


# ============================================================================
# Ausführung
# ============================================================================

def main():
    """Hauptfunktion mit Benutzerinteraktion"""
    global SCENARIOS, PARALLEL_TRAINING

    print("=" * 60)
    print("MULTI-SZENARIO Q-LEARNING TRAINER")
    print("=" * 60)

    # Interaktive Konfiguration
    configure_training_interactively()
    SCENARIOS = select_scenarios_interactively()

    mode = input(f"\nTraining-Modus - (s)equenziell oder (p)arallel? (s): ").strip().lower()
    PARALLEL_TRAINING = (mode == 'p')

    # Bestätigung mit erweiterten Informationen
    print(f"\n📋 TRAINING KONFIGURATION:")
    print(f"  Szenarien: {list(SCENARIOS.keys())}")
    print(f"  Modus: {'Parallel' if PARALLEL_TRAINING else 'Sequenziell'}")
    print(f"  Visualisierungen: {'Interaktiv' if SHOW_VISUALIZATIONS else 'Automatisiert'}")
    print(f"  Parameter werden aus config.py geladen")

    if SHOW_VISUALIZATIONS:
        print(f"  ⚠️  Sie müssen {len(SCENARIOS) * 3} Matplotlib-Fenster schließen!")
        print(f"      (3 Diagramme pro Szenario)")
    else:
        print(f"  ✅ Läuft vollautomatisch durch")

    # Geschätzte Zeit basierend auf config.py
    config_params = load_current_config()
    estimated_time = len(SCENARIOS) * (config_params['EPISODES'] / 1000) * 2
    print(f"  ⏱️  Geschätzte Dauer: {estimated_time:.0f}-{estimated_time * 2:.0f} Minuten")

    confirm = input(f"\nTraining starten? (j/n): ").strip().lower()

    if confirm == 'j':
        train_all_scenarios()
    else:
        print("Training abgebrochen.")


if __name__ == "__main__":
    main()