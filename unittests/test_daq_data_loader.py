import unittest
from packages.daq_data_loader import *
from pathlib import Path


class DAQLoaderTests(unittest.TestCase):
    def setUp(self):
        self.loader = DAQLoader(test_mode=True)
        self.home_dir = self.loader.home_dir

    def test_check_if_valid_folder(self):
        test_path = self.loader.saves_path / "QcWatt_2025-04-12_18_19_34"
        self.assertTrue(self.loader.check_if_valid_folder(test_path))

    def test_get_save_folders(self):
        folders = self.loader.get_save_folders()
        expected_folders = [self.loader.saves_path / 'QcWatt_2025-04-12_18_19_22', self.loader.saves_path / 'QcWatt_2025-04-12_18_19_34']
        self.assertEqual(folders, expected_folders)

    def test_load_save_w_invalid_index(self):
        self.assertRaises(IndexError, self.loader.load_save, -1)
        self.assertRaises(IndexError, self.loader.load_save, 2)

