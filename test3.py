import numpy as np
from pathlib import Path

home = Path(__file__).parents[0]
saves_path = home / "Saves"
current_file_path = saves_path / "QcWatt_2025-04-22_23_42_57"
time_npy_path = current_file_path / "time.npy"
time_array = np.load(time_npy_path)
tension_npy_path = current_file_path / "tension.npy"
tension_array = np.load(tension_npy_path)
bits_npy_path = current_file_path / "bits.npy"
bits_array = np.load(bits_npy_path)
print(time_array)
# print(tension_array.shape)
# print(bits_array.shape)