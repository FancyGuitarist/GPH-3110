import numpy as np
from pathlib import Path
import pandas as pd
import unittest
import sys


class DAQLoader:
    def __init__(self):
        self.home_dir = Path(__file__).parents[1]
        self.saves_path = self.home_dir / 'Saves'
        self.required_stems = ["bits", "tension", "time"]
        self.save_folders = self.get_save_folders()

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