import unittest
from packages.daq_data_loader import *


class DAQLoaderTests(unittest.TestCase):
    def setUp(self):
        self.loader = DAQLoader()

    def test_check_if_valid_folder(self):
        test_path = self.loader.saves_path / "QcWatt_2025-04-11_900nm"
        self.assertTrue(self.loader.check_if_valid_folder(test_path))

    def test_get_folders(self):
        folders = self.loader.get_save_folders()

