from enum import StrEnum
import pandas as pd
from pathlib import Path
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.optimize as opt

transmission = 0.1
home_directory = Path(__file__).parents[1]
T_values = [
    297.800093871208,
    297.795935403683,
    300,
    297.793461919386,
    297.797705629502,
    297.801065549678,
]


def gaussian_2d(coords: tuple, A, x0, y0, sigma_x, sigma_y, T_0):
    x, y = coords
    return (
        A
        * np.exp(-((x - x0) ** 2 / (2 * sigma_x**2) + (y - y0) ** 2 / (2 * sigma_y**2)))
        + T_0
    )


class Thermistance:
    def __init__(
        self,
        position: tuple,
        port: str,
    ):
        """
        Thermistance class to represent the thermistances in the power meter.
        :param position: Position in polar coordinates (r, theta) with the origin at the center of the textured plate.
        :param port: Port of the thermistance on the NI-DAQ.
        """
        self.position = position
        self.port = port
        self.steinhart_coeffs = (1.844e-3, -3.577e-06, 2.7612e-05, -1.0234e-06)
        self.demo_mode = True

    def __repr__(self):
        return f"Position: {self.position[0]}e^{self.position[1]}j, Port: {self.port}"

    @property
    def x(self):
        return self.position[0] * np.cos(self.position[1])

    @property
    def y(self):
        return self.position[0] * np.sin(self.position[1])

    @property
    def r(self):
        return self.position[0]

    @property
    def theta(self):
        return self.position[1]

    def get_tension(self):
        match self.port:
            case "port0" | "port1" | "port2":
                return 12 * random.random()
            case "port3" | "port4" | "port5":
                return 10 * random.random()

    def get_resistance(self):
        G_w = self.get_tension() / 5  # 5 is the tension of the power supply
        return (130 - 230 * G_w) / (10 + 23 * G_w)

    def get_temperature(self):
        if not self.demo_mode:
            R_t = self.get_resistance()
            R_m = 240  # Intrinsic resistance of the multiplexor
            A, B, C, D = self.steinhart_coeffs
            T_inv = (
                A
                + np.log(R_t - R_m) * B
                + np.log(R_t - R_m) ** 2 * C
                + np.log(R_t - R_m) ** 3 * D
            )
            return 1 / T_inv - 273.15  # Convert to Celsius
        else:
            return T_values[int(self.port[-1])]


class GlassType(StrEnum):
    VG9 = "VG9"
    KG2 = "KG2"
    NG11 = "NG11"


class Glass:
    def __init__(self, glass_type: GlassType):
        self.glass_type = glass_type
        self.transmission_values_cache = None
        self.wavelength_values_cache = None
        self.spectrum_cache = None

    @property
    def n_properties(self):
        properties = {}
        match self.glass_type:
            case GlassType.VG9:
                properties["a"] = [1.1839, 0.00763]
                properties["b"] = [0.0336, 0.043272]
                properties["c"] = [1.111, 116.448]
            case GlassType.KG2:
                properties["a"] = [1.171719, 0.006324]
                properties["b"] = [0.097958, 0.031092]
                properties["c"] = [0.071306, 10.066]
            case GlassType.NG11:
                properties["a"] = [0.3483, 0.01326]
                properties["b"] = [1.0034, 0.012265]
                properties["c"] = [34.8247, 5797.735]
        return properties

    @property
    def transmission_spectrum(self):
        if self.spectrum_cache is None:
            match self.glass_type:
                case GlassType.VG9:
                    self.spectrum_cache = pd.read_csv(
                        Path(home_directory, "Glass_Spectrums", "VG9.txt")
                    )
                case GlassType.KG2:
                    self.spectrum_cache = pd.read_csv(
                        Path(home_directory, "Glass_Spectrums", "KG2.csv")
                    )
                case GlassType.NG11:
                    self.spectrum_cache = pd.read_csv(
                        Path(home_directory, "Glass_Spectrums", "NG11.csv")
                    )
        return self.spectrum_cache

    @property
    def transmission_values(self):
        if self.transmission_values_cache is None:
            self.transmission_values_cache = self.transmission_spectrum[
                "Transmission"
            ].to_numpy()
        return self.transmission_values_cache

    @property
    def wavelength_values(self):
        if self.wavelength_values_cache is None:
            self.wavelength_values_cache = self.transmission_spectrum[
                "Wavelength"
            ].to_numpy()
        return self.wavelength_values_cache

    def get_potential_wavelengths(
        self, read_transmission: float, interval: float = 0.1
    ):
        candidates_upper = self.transmission_values < read_transmission + (
            interval * read_transmission
        )
        candidates_lower = self.transmission_values > read_transmission - (
            interval * read_transmission
        )
        return self.wavelength_values[candidates_upper & candidates_lower]

    def show_spectrum(self):
        plt.plot(
            self.wavelength_values, self.transmission_values, label=self.glass_type
        )
        plt.xlim(200, 2500)
        plt.ylim(0, 100)
        plt.show()


class PowerMeter:
    def __init__(self):
        self.glasses = [
            Glass(GlassType.NG11),
            Glass(GlassType.KG2),
            Glass(GlassType.VG9),
        ]
        self.thermistances = self.setup_thermistance_grid()
        self.laser_initial_guesses = [10, 0, 0, 1, 1, 290]
        self.laser_params = None

    @property
    def x_coords(self):
        return np.array([t.x for t in self.thermistances])

    @property
    def y_coords(self):
        return np.array([t.y for t in self.thermistances])

    @property
    def xy_coords(self):
        return self.x_coords, self.y_coords

    def setup_thermistance_grid(self):
        self.thermistances = []
        for i in range(6):
            self.thermistances.append(
                Thermistance((0.550, np.pi / 3 + i * np.pi / 3), f"port{i}")
            )
        return self.thermistances

    def get_temperature_values(self):
        return [thermistance.get_temperature() for thermistance in self.thermistances]

    def get_laser_params(self):
        self.laser_params, _ = opt.curve_fit(
            gaussian_2d,
            self.xy_coords,
            self.get_temperature_values(),
            p0=self.laser_initial_guesses,
            maxfev=1000,
        )
        return self.laser_params

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

    def get_laser_position(self, temps: list):
        pass

    def get_incident_power(self, temps: list):
        pass

    def estimate_absorbance_of_glass(self, temps: list, glass_type: Glass):
        pass

    def estimate_wavelength(self):
        pass


if __name__ == "__main__":
    pass
