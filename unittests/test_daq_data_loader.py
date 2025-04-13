import unittest

import numpy as np

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

    def test_load_save_cache(self):
        self.assertEqual(self.loader.load_cache, None)
        self.loader.load_save(1)
        self.assertFalse(self.loader.load_cache is None)

    def test_get_combobox_options(self):
        expected_combobox_options = ['2025-04-12_18_19_22', '2025-04-12_18_19_34']
        combobox_options = self.loader.get_combobox_options()
        self.assertEqual(combobox_options, expected_combobox_options)

    def test_load_save_for_ui(self):
        _, full_time_array, full_tension_array, full_bits_array = self.loader.load_save(0)
        time_vals_1, tension_vals_1, bits_vals_1 = self.loader.load_save_for_ui(0)
        self.assertEqual(time_vals_1.shape, full_time_array[:0, :].shape)
        self.assertEqual(tension_vals_1.shape, full_tension_array[:0, :].shape)
        self.assertEqual(bits_vals_1.shape, full_bits_array[:0].shape)
        time_vals_2, tension_vals_2, bits_vals_2 = self.loader.load_save_for_ui(0)
        self.assertEqual(time_vals_2.shape, full_time_array[:16, :].shape)
        self.assertEqual(tension_vals_2.shape, full_tension_array[:16, :].shape)
        self.assertEqual(bits_vals_2.shape, full_bits_array[:16].shape)
        time_vals_3, tension_vals_3, bits_vals_3 = self.loader.load_save_for_ui(0)
        self.assertEqual(time_vals_3.shape, full_time_array[:32, :].shape)
        self.assertEqual(tension_vals_3.shape, full_tension_array[:32, :].shape)
        self.assertEqual(bits_vals_3.shape, full_bits_array[:32].shape)

