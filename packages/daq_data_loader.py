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
        self.load_index = 0
        self.test_mode = test_mode

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

    def load_save_for_ui(self, index: int):
        _, time, tension, bits = self.load_save(index)
        current_time = time[:self.load_index, :]
        current_tension = tension[:self.load_index, :]
        current_bit = bits[:self.load_index]
        if self.load_index == bits.shape[0] - 1:
            self.load_index = 0
        elif 16 > bits.shape[0] - self.load_index:
            self.load_index = bits.shape[0] - 1
        else:
            self.load_index += 16
        return current_time, current_tension, current_bit




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