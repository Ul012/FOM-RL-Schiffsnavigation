# utils/evaluation_export.py

import csv
import os

def export_results_to_csv(results, output_path="exports/evaluation_summary.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Szenario", "Success Rate (%)", "Total Reward", "Avg. Reward", "Avg. Steps", "Reward Variance"])
        for result in results:
            writer.writerow([
                result.get("name", ""),
                result.get("success_rate", ""),
                result.get("total_reward", ""),
                result.get("avg_reward", ""),
                result.get("avg_steps", ""),
                result.get("reward_variance", "")
            ])
    print(f"📄 CSV-Datei erfolgreich gespeichert unter: {output_path}")
