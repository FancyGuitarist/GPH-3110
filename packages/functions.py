from enum import StrEnum
import numpy as np
import random

transmission = 0.1


class Thermistance:
    def __init__(
        self,
        position: tuple,
        port: str,
        steinhart_coeffs: tuple = (1.844e-3, -3.577e-06, 2.7612e-05, -1.0234e-06),
    ):
        self.position = position
        self.port = port
        self.steinhart_coeffs = steinhart_coeffs

    def __repr__(self):
        return f"Position: {self.position}, Port: {self.port}"

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    def get_tension(self):
        match self.port:
            case "port0" | "port1" | "port2":
                return 253 + (20 * random.random())
            case "port3" | "port4" | "port5":
                return 293 + (20 * random.random())

    def get_resistance(self):
        G_w = self.get_tension() / 5  # 5 is the tension of the power supply
        return (130 - 230 * G_w) / (10 + 23 * G_w)

    def get_temperature(self):
        R_t = self.get_resistance()
        A, B, C, D = self.steinhart_coeffs
        T_inv = A + np.log(R_t) * B + np.log(R_t) ** 2 * C + np.log(R_t) ** 3 * D
        return 1 / T_inv - 273.15  # Convert to Celsius


class Glasses(StrEnum):
    VG9 = "VG9"
    KG2 = "KG2"
    NG11 = "NG11"

    @property
    def n_properties(self):
        properties = {}
        match self:
            case Glasses.VG9:
                properties["a"] = [1.1839, 0.00763]
                properties["b"] = [0.0336, 0.043272]
                properties["c"] = [1.111, 116.448]
            case Glasses.KG2:
                properties["a"] = [1.171719, 0.006324]
                properties["b"] = [0.097958, 0.031092]
                properties["c"] = [0.071306, 10.066]
            case Glasses.NG11:
                properties["a"] = [0.3483, 0.01326]
                properties["b"] = [1.0034, 0.012265]
                properties["c"] = [34.8247, 5797.735]
        return properties


class PlateProperties(StrEnum):
    Laserax = "Laserax"

    @property
    def property(self):
        if self == PlateProperties.Laserax:
            return 0.1


class PowerMeter:
    def __init__(self, thermistances: list[Thermistance]):
        self.voltage = 0
        self.temperature = 0
        self.incident_power = 0
        self.absorbance = 0
        self.wavelength = 0
        self.glasses = [Glasses.NG11, Glasses.KG2, Glasses.VG9]
        self.thermistances = thermistances

    def n_glasses(self, lambda_: float):
        ns = []
        for glass in self.glasses:
            a_coeffs, b_coeffs, c_coeffs = (
                glass.n_properties["a"],
                glass.n_properties["b"],
                glass.n_properties["c"],
            )
            a = a_coeffs[0] * lambda_**2 / (lambda_**2 - a_coeffs[1])
            b = b_coeffs[0] * lambda_**2 / (lambda_**2 - b_coeffs[1])
            c = c_coeffs[0] * lambda_**2 / (lambda_**2 - c_coeffs[1])
            ns.append(np.sqrt(1 + a + b + c))
        return ns

    def convert_voltage_to_temp(self, voltage: float):
        pass

    def get_laser_position(self, temps: list):
        pass

    def get_incident_power(self, temps: list):
        pass

    def estimate_absorbance_of_glass(self, temps: list, glass_type: Glasses):
        pass

    def estimate_wavelength(self, glasses: list[Glasses]):
        pass
