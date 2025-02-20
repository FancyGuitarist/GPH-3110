from enum import StrEnum


transmission = 0.1


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

    def estimateAbsorbanceOfGlass(self, temps: list, glass_type: Glasses):
        pass

    def estimateWaveLength(self, glasses: list[Glasses]):
        pass
