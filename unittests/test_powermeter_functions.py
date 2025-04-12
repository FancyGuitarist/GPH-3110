import unittest
from packages.powermeter_functions import PowerMeter
import numpy as np


class PowerMeterTests(unittest.TestCase):
    @unittest.skip("Have to update test for new implementations")
    def test_thermistance_grid(self):
        power_meter = PowerMeter()
        self.assertEqual(len(power_meter.thermistances), 9)
        index = 0
        for port, thermistance in power_meter.thermistances.items():
            self.assertEqual(thermistance.r, 0.550)
            self.assertEqual(thermistance.theta, np.pi / 3 + (index * np.pi / 3))
            index += 1
