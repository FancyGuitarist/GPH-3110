import numpy as np
from pathlib import Path
import pandas as pd


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

if __name__ == '__main__':
    test_loader = DAQLoader()
    print(test_loader.get_save_folders())