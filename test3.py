import numpy as np
from pathlib import Path

home = Path(__file__).parents[0]
saves_path = home / "Saves"
time_npy_path = saves_path / "QcWatt_2025-04-18_10_04_41" / "tension.npy"
time_array = np.load(time_npy_path)
print(time_array)