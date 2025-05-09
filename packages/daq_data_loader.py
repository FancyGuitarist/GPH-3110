import numpy as np
from pathlib import Path
import pandas as pd
import unittest
import sys


class DAQLoader:
    def __init__(self, test_mode=False):
        self.home_dir = Path(__file__).parents[1]
        self.saves_path = self.home_dir / 'Saves' if not test_mode else self.home_dir / 'Saves' / 'test_saves'
        self.required_stems = ["bits", "tension", "time"]
        self.save_folders = self.get_save_folders()
        self.combobox_options = self.get_combobox_options()
        self.load_cache = None
        self.load_index = 1
        self.test_mode = test_mode
        self.save_index = 0

    def check_if_valid_folder(self, folder: Path):
        file_stems = [path.stem for path in folder.rglob('*.npy')]
        return file_stems == self.required_stems

    def get_save_folders(self):
        self.save_folders = []
        folders = list(self.saves_path.iterdir())
        for folder in folders:
            if self.check_if_valid_folder(folder):
                self.save_folders.append(folder)
        return self.save_folders

    def get_combobox_options(self):
        self.combobox_options = []
        for folder in self.save_folders:
            self.combobox_options.append(str(folder.name).replace("QcWatt_", ""))
        return self.combobox_options

    def find_combobox_index(self, combobox_option):
        return self.combobox_options.index(combobox_option)

    def load_save(self, index: int):
        if self.load_cache is None or self.load_cache[0] != index:
            if index > len(self.save_folders) or index < 0:
                raise IndexError("Wrong index, current list is {} long".format(self.save_folders))
            folder = self.save_folders[index]
            time = np.load(folder / 'time.npy')
            tension = np.load(folder / 'tension.npy')
            bits = np.load(folder / 'bits.npy')
            self.load_cache = (index, time, tension, bits)
        return self.load_cache

    def set_save_index(self, save_index: int):
        self.save_index = save_index
        return self.save_index

    def load_save_for_ui(self, looping=False):
        _, time, tension, bits = self.load_save(self.save_index)

        # Ensure data exists
        if bits.shape[0] == 0:
            print("Error: No data in loaded save.")
            return None  # Or handle gracefully

        if self.load_index >= bits.shape[0]:  # Prevent index out-of-bounds
            self.load_index = bits.shape[0] - 1

        current_time = time[self.load_index - 1:self.load_index, 0]
        current_tension = tension[self.load_index - 1:self.load_index, :]
        current_bit = bits[self.load_index - 1:self.load_index]

        if self.load_index == bits.shape[0] - 1:
            self.load_index = 1 if looping else bits.shape[0] - 1
        else:
            self.load_index += 1
        print(current_time[0], current_tension[0], current_bit[0])
        return current_time[0], current_tension[0], current_bit[0]




# Add test folder path to sys.path
test_folder_path = Path(__file__).parents[1] / 'unittests'
sys.path.insert(0, str(test_folder_path))

def run_tests():
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(test_folder_path), pattern="test_*.py")  # Adjust your test file naming pattern if needed
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == "__main__":
    run_tests()