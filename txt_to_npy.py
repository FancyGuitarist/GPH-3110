import numpy as np
import pandas as pd
from pathlib import Path

# Short script to convert lab data to the same format as the DAQLoader class

home_directory = Path(__file__).parents[0]
lab_paths = home_directory / "lab_files"
files = lab_paths.glob("*.txt")
for file in files:
    print(file)
    save_folder_path = home_directory / "Saves"
    save_path = save_folder_path / f"QcWatt_{str(file.stem)}"
    save_path.mkdir(parents=True, exist_ok=True)
    save_path_time = save_path / "time.npy"
    save_path_tension = save_path / "tension.npy"
    save_path_bits = save_path / "bits.npy"


    names_to_modify = ["TH1 data", "TH2 data", "TH3 data", "THmux1_list", "THmux2_list"]
    names_to_use = ["TH1", "TH2", "TH3", "THmux1", "THmux2"]
    file_data = open(file, 'r').read()
    for index, name in enumerate(names_to_use):
        file_data = file_data.replace(names_to_modify[index], name)

    with open(file, 'w') as f:
        f.write(file_data)

    df = pd.read_csv(file, sep=r"\s+", header=0, skiprows=[1])
    time_array = np.tile(df['Time'].to_numpy()[:, np.newaxis], (1, 5))
    tension_array = np.array([df['TH1'].to_numpy(), df['TH2'].to_numpy(), df['TH3'].to_numpy(), df['THmux1'].to_numpy(), df['THmux2'].to_numpy()]).T
    bits_array = np.array([x for x in range(16)])
    bits_array = np.tile(bits_array, time_array.shape[0] // 16 + 1)[:time_array.shape[0]]

    print(time_array.shape, tension_array.shape, bits_array.shape)

    np.save(save_path_time, time_array)
    np.save(save_path_tension, tension_array)
    np.save(save_path_bits, bits_array)