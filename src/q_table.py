import numpy as np

# Lade die Q-Tabelle
q_table = np.load("q_table.npy")

# Gib einen Teil davon aus (z. B. die ersten 5 Einträge)
print(q_table[:5])
