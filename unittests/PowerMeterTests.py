import unittest
from packages.functions import PowerMeter
import numpy as np


class PowerMeterTests(unittest.TestCase):
    def test_thermistance_grid(self):
        power_meter = PowerMeter()
        self.assertEqual(len(power_meter.thermistances), 6)
        print(power_meter.thermistances)
        for index, thermistance in enumerate(power_meter.thermistances):
            self.assertEqual(thermistance.r, 0.550)
            self.assertEqual(thermistance.theta, np.pi / 3 + (index * np.pi / 3))
