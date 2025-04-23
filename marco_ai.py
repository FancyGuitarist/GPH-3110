import numpy as np
from skimage.draw import disk
from PIL import Image
from pathlib import Path
import pandas as pd
from packages.daq_data_loader import DAQLoader
from packages.powermeter_functions import DAQPort, Thermistor, PowerMeter
import matplotlib.pyplot as plt

file_to_load = "QcWatt_2025-04-18_09_53_57"
file_to_load = file_to_load.replace("QcWatt_", "")

r_out, r_int = 13.97, 5
powermeter = PowerMeter()
calibration_arrays = powermeter.calibration_arrays

glass_angles = [(n * (2 * np.pi)/3) + np.pi/2 for n in range(3)]

glass_1_ports = [DAQPort("4.4"), DAQPort("4.8"), DAQPort("4.0")]
glass_1_ref_port = DAQPort("5.6")

glass_1_thermistors = {}
for index, port in enumerate(glass_1_ports):
    angle = glass_angles[index]
    glass_1_thermistors[port] = Thermistor((r_int, angle), port, calibration_arrays)

glass_1_thermistors[glass_1_ref_port] = Thermistor((0, 0), glass_1_ref_port, calibration_arrays)

# glass_2_ports = [DAQPort("4.1"), DAQPort("4.6"), DAQPort("4.14")]
# glass_2_ref_port = DAQPort("5.12")
glass_3_ports = [DAQPort("4.2"), DAQPort("4.10"), DAQPort("4.12")]
glass_3_ref_port = DAQPort("5.2")

glass_3_thermistors = {}
for index, port in enumerate(glass_3_ports):
    angle = glass_angles[index]
    glass_3_thermistors[port] = Thermistor((r_int, angle), port, calibration_arrays)

glass_3_thermistors[glass_3_ref_port] = Thermistor((0, 0), glass_3_ref_port, calibration_arrays)

out_ports = [DAQPort("5.13"), DAQPort("5.3"), DAQPort("5.11"), DAQPort("5.7"), DAQPort("5.9"), DAQPort("5.5")]
in_ports = [DAQPort("1"), DAQPort("2"), DAQPort("3")]
plate_ref_port = DAQPort("5.1")
plate_ports = out_ports + in_ports + [plate_ref_port]
plate_thermistors = {}
angles_list = []

for i in range(6):
    angle = i * np.pi / 3 + np.pi/6
    angles_list.append(angle)
    plate_thermistors[out_ports[i]] = Thermistor((r_out, angle), out_ports[i], calibration_arrays)

plate_thermistors[DAQPort("1")] = Thermistor((r_int, angles_list[5]), DAQPort("1"), calibration_arrays)
plate_thermistors[DAQPort("2")] = Thermistor((r_int, angles_list[1]), DAQPort("2"), calibration_arrays)
plate_thermistors[DAQPort("3")] = Thermistor((r_int, angles_list[3]), DAQPort("3"), calibration_arrays)
plate_thermistors[plate_ref_port] = Thermistor((0, 0), plate_ref_port, calibration_arrays)


loader = DAQLoader()
save_folders = loader.get_save_folders()
index_to_load = loader.find_combobox_index(file_to_load)
print([path.stem for path in save_folders])

save = list(loader.load_save(index_to_load))
print(save)
save = save[1:]
save[0] = save[0][:, 0].tolist()
save[1] = np.swapaxes(save[1], 0, 1).tolist()
save[2] = save[2].tolist()
max_len = len(save[0])
save = tuple(save)
# print(len(save[1][0]))
powermeter_thermistors = {}
powermeter_thermistors.update(plate_thermistors)
powermeter_thermistors.update(glass_1_thermistors)
powermeter_thermistors.update(glass_3_thermistors)

# print([t for t in plate_thermistors.keys()])
# print([t for t in glass_1_thermistors.keys()])
# print([t for t in glass_3_thermistors.keys()])
# print([t for t in powermeter_thermistors.keys()])

powermeter_temperatures = {"Time": np.array(save[0])}
for port, thermistor in powermeter_thermistors.items():
    port_values = powermeter.get_port_values(port, save)
    # print(np.mean(np.diff(port_values[0])))
    thermistor.add_data(port_values)
    if len(port_values[0]) < max_len:
        initial_temps = thermistor.get_temperature(mean=False)
        temps = np.repeat(initial_temps, 16)[:max_len]
        # temps = np.zeros(max_len, dtype=initial_temps.dtype)
        # temps[::16] = initial_temps
    else:
        temps = thermistor.get_temperature(mean=False)
    powermeter_temperatures[port] = temps[:4464]

print([t.shape for t in powermeter_temperatures.values()])
time = powermeter_temperatures["Time"][:4464]
plt.figure(figsize=(10, 6))

# Plot each port
for port, temps in powermeter_temperatures.items():
    if port != "Time":
        plt.plot(time, temps, label=port)

# Add labels, title, legend
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.title("Temperature Over Time for Each Port")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# print(powermeter_temperatures[DAQPort("5.13")].shape)
# print(powermeter_temperatures[DAQPort("1")].shape)




