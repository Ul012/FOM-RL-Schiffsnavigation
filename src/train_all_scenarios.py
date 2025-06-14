# train_all_scenarios.py

# ============================================================================
# Imports
# ============================================================================

import sys
import os
import time
import subprocess
from datetime import datetime

# Projektstruktur für Import anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from utils.common import setup_export

# ============================================================================
# Konfiguration
# ============================================================================

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

SHOW_VISUALIZATIONS = False
PARALLEL_TRAINING = False

# ============================================================================
# Ausführung des Trainings für ein einzelnes Szenario
# ============================================================================

def run_training_for_scenario(scenario_name, scenario_config):
    print(f"\n{'=' * 60}")
    print(f"STARTE TRAINING: {scenario_name.upper()}")
    print(f"{'=' * 60}")
    print(f"Beschreibung: {scenario_config['description']}")
    print(f"Modus: {scenario_config['env_mode']}")
    print(f"Umgebung: {scenario_config['environment']}")

    start_time = time.time()

    try:
        env = os.environ.copy()
        env["ENV_MODE"] = scenario_config["env_mode"]
        env["EXPORT_PDF"] = "False" if not SHOW_VISUALIZATIONS else "True"
        env["SHOW_VISUALIZATIONS"] = "False" if not SHOW_VISUALIZATIONS else "True"

        result = subprocess.run(
            [sys.executable, "train.py"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=1800,
            env=env
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ Training erfolgreich abgeschlossen ({duration:.1f}s)")
            print(f"Q-Tabelle gespeichert: q_table_{scenario_config['env_mode']}.npy")
            output_lines = result.stdout.split('\n') # wenn Ausgabe kürzer sein soll:[-20:] anhängen direkt hinter ('\n')
            for line in output_lines:
                if line.strip():
                    print(f"  {line}")
        else:
            print(f"❌ Training fehlgeschlagen ({duration:.1f}s)")
            print(f"Fehler: {result.stderr}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"⏰ Training-Timeout nach 30min")
        return False
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        return False

# ============================================================================
# Ausführung
# ============================================================================

def train_all_scenarios():
    print("🏗️  MULTI-SZENARIO TRAINING")
    print(f"Anzahl Szenarien: {len(SCENARIOS)}")
    print(f"Training-Modus: {'Parallel' if PARALLEL_TRAINING else 'Sequenziell'}")

    setup_export()

    if PARALLEL_TRAINING:
        print("⚠️  Parallel Training ist derzeit deaktiviert")
        return
    else:
        results = {}
        for i, (scenario_name, scenario_config) in enumerate(SCENARIOS.items(), 1):
            print(f"\n[{i}/{len(SCENARIOS)}] Nächstes Szenario: {scenario_name}")
            results[scenario_name] = run_training_for_scenario(scenario_name, scenario_config)
            time.sleep(2)

    print(f"\n{'=' * 60}")
    print("TRAINING ZUSAMMENFASSUNG")
    print(f"{'=' * 60}")

    for scenario_name, success in results.items():
        status = "✅ Erfolgreich" if success else "❌ Fehlgeschlagen"
        q_table_exists = os.path.exists(f"q_table_{SCENARIOS[scenario_name]['env_mode']}.npy")
        q_table_status = "Q-Tabelle ✓" if q_table_exists else "Q-Tabelle ✗"
        print(f"{scenario_name:<20} {status:<15} {q_table_status}")

if __name__ == "__main__":
    train_all_scenarios()
