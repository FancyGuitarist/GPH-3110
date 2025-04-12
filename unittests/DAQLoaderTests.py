import unittest
from packages.daq_data_loader import *
from pathlib import Path


class DAQLoaderTests(unittest.TestCase):
    def setUp(self):
        self.loader = DAQLoader()
        self.home_dir = self.loader.home_dir

    def test_check_if_valid_folder(self):
        test_path = self.loader.saves_path / "QcWatt_2025-04-04_941"
        self.assertTrue(self.loader.check_if_valid_folder(test_path))

    def test_get_folders(self):
        folders = self.loader.get_save_folders()
        expected_folders = [self.loader.saves_path / 'QcWatt_2025-04-04_941', self.loader.saves_path / 'QcWatt_2025-04-07_993', self.loader.saves_path / 'QcWatt_2025-04-07_994']
        self.assertEqual(folders, expected_folders)

