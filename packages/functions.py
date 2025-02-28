from enum import StrEnum
import random


transmission = 0.1


class Thermistance:
    def __init__(self, position: tuple, port: str):
        self.position = position
        self.port = port

    def __repr__(self):
        return f"Position: {self.position}, Port: {self.port}"

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    def get_temperature(self):
        match self.port:
            case "port0" | "port1" | "port2":
                return 253 + (20 * random.random())
            case "port3" | "port4" | "port5":
                return 293 + (20 * random.random())


class Glasses(StrEnum):
    Glass1 = "Glass1"

    @property
    def property(self):
        if self == Glasses.Glass1:
            return 0.1


class PlateProperties(StrEnum):
    Laserax = "Laserax"

    @property
    def property(self):
        if self == PlateProperties.Laserax:
            return 0.1


class PowerMeter:
    def __init__(self):
        self.voltage = 0
        self.temperature = 0
        self.incident_power = 0
        self.absorbance = 0
        self.wavelength = 0
        self.glasses = [Glasses.Glass1]

    def convertVoltageToTemp(self, voltage: float):
        pass

    def findMaxTempPosition(self, temps: list):
        pass

    def findIncidentPower(self, temps: list):
        pass
    def estimateAbsorbanceOfGlass(self, temps: list, glass_type : Glasses):
        pass

    def estimateWaveLength(self, glasses: list[Glasses]):
        pass