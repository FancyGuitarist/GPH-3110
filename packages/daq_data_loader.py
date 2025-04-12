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

    def load_save(self, index: int):
        if index > len(self.save_folders) or index < 0:
            raise IndexError("Wrong index, current list is {} long".format(self.save_folders))
        folder = self.save_folders[index]
        time = np.load(folder / 'time.npy')
        tension = np.load(folder / 'tension.npy')
        bits = np.load(folder / 'bits.npy')
        return time, tension, bits



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