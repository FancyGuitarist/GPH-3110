from enum import StrEnum
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sys
import time
import datetime
import re
from packages.daq_data_loader import DAQLoader


class AppModes(StrEnum):
    """
    Enum class to define the app modes.
    """
    open = "open"
    demo = "demo"
    locked = "locked"


if sys.platform == "win32":
    import nidaqmx
    from nidaqmx.system import System
    from nidaqmx.stream_readers import AnalogMultiChannelReader
    from nidaqmx.constants import AcquisitionType, LineGrouping, TerminalConfiguration

    app_mode = AppModes.open
else:
    app_mode = AppModes.locked

transmission = 0.1
home_directory = Path(__file__).parents[1]


def gaussian_2d(coords: tuple, A, x0, y0, sigma_x, sigma_y):
    x, y = coords
    # return A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2)/ (2 * sigma ** 2))
    return A * np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))


class DAQPort:
    def __init__(self, port: str):
        self.port_str = port

    def __str__(self):
        return self.port_str

    def __repr__(self):
        return self.port_str

    @property
    def daq_port(self):
        return int(re.search(r'^(\d+)(?:\.(\d+))?$', self.port_str).group(1))

    @property
    def demux_port(self):
        demux_port = re.search(r'^(\d+)(?:\.(\d+))?$', self.port_str).group(2)
        return int(demux_port) + 1 if demux_port else None

    def __eq__(self, other):
        if isinstance(other, DAQPort):
            return self.port_str == other.port_str
        return False

    def __hash__(self):
        return hash(self.port_str)

    def get_transfer_function_params(self):
        params_dict = {"V_s": 4.98, "R_m": 0}
        if 0 <= self.daq_port <= 3:
            params_dict["G_a"] = 6.495130137713433
            params_dict["G_ap"] = 2.2137861972053634
            params_dict["X_c"] = 0.8675482699505241
        elif self.daq_port == 4:
            params_dict["G_a"] = 4.6513235468240754
            params_dict["G_ap"] = 1.6160213093655558
            params_dict["X_c"] = 1.1695513675653928
        elif self.daq_port == 5:
            params_dict["G_a"] = 6.4364552626488525
            params_dict["G_ap"] = 2.293894448811315
            params_dict["X_c"] = 0.8383478557216358
        else:
            raise ValueError("Enter a valid DAQ port number")
        return params_dict


class Thermistance:
    def __init__(
            self,
            position: tuple,
            port: DAQPort,
            calibration_arrays: list[np.ndarray],
    ):
        """
        Thermistance class to represent the thermistances in the power meter.
        :param position: Position in polar coordinates (r, theta) with the origin at the center of the textured plate.
        :param port: DAQPort of the thermistance on the NI-DAQ.
        """
        self.position = position
        self.port = port
        self.steinhart_coeffs = (1.844e-3, -3.577e-06, 2.7612e-05, -1.0234e-06)
        self.calibration_arrays = calibration_arrays
        self.data = None

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

    @property
    def no_data(self):
        return True if self.data is None else False

    def add_data(self, port_data):
        self.data = port_data

    def get_calibration_data(self):
        if 1 <= self.port.daq_port <= 3:
            calibration_data = self.calibration_arrays[0]
        elif self.port.daq_port == 4:
            calibration_data = self.calibration_arrays[2]
        elif self.port.daq_port == 5:
            calibration_data = self.calibration_arrays[1]
        else:
            raise ValueError("Enter a valid DAQ port number")
        return calibration_data

    def split_extrapolation_array(self, V_m):
        cutoff_val = self.get_calibration_data()[1].max()
        calibration_mask = V_m <= cutoff_val
        V_m_calibration = V_m[calibration_mask]
        V_m_transfer_func = V_m[~calibration_mask]
        return V_m_calibration, V_m_transfer_func

    def extrapolate_w_transfer_function(self, V_m):
        params_dict = self.port.get_transfer_function_params()
        A, B, C, D = self.steinhart_coeffs
        V_s, R_m = params_dict['V_s'], params_dict['R_m']
        G_a, G_ap, X_c = params_dict['G_a'], params_dict['G_ap'], params_dict['X_c']
        V_m = (V_m - G_a * X_c + G_ap * X_c) / G_ap
        x = np.log(1000 * ((130 - 230 * (V_m / V_s)) / (10 + 23 * (V_m / V_s)) - R_m))
        return 1 / (A + B * x + C * x ** 2 + D * x ** 3)

    def extrapolate_w_calibration_data(self, V_m):
        calibration_data = self.get_calibration_data()
        return np.interp(V_m, calibration_data[:, 1], calibration_data[:, 0])

    def get_temperature(self):
        if self.no_data:
            return 0
        V_m = self.data[1]
        V_m_calibration, V_m_transfer_func = self.split_extrapolation_array(V_m)
        temp_calibration = self.extrapolate_w_calibration_data(V_m_calibration)
        if not np.any(V_m_transfer_func):
            temp_transfer_func = np.array([])
        else:
            temp_transfer_func = self.extrapolate_w_transfer_function(V_m_transfer_func)
        temp = np.hstack([temp_calibration, temp_transfer_func])
        return np.mean(temp)


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
    def __init__(self, samples_per_read: int = 10, sample_rate=10000):
        self.samples_per_read = samples_per_read
        self.sample_rate = sample_rate

        self.glasses = [
            Glass(GlassType.NG11),
            Glass(GlassType.KG2),
            Glass(GlassType.VG9),
        ]
        self.r_int = 5  # mm
        self.r_out = 13.97  # mm
        self.calibration_arrays = self.load_calibration_files()
        self.thermistances, self.ports = self.setup_thermistance_grid()
        self.bits_list = self.setup_bits_list()
        self.laser_initial_guesses = [10, 0, 0, 8.5, 8.5]
        self.laser_params = None
        self.loading_mode = False
        self.manual_wavelength = None
        self.task, self.reader, self.do_task, self.start_time, self.data = None, None, None, None, None
        self.i = 0
        self.time_cache, self.tension_cache = [[] for _ in range(5)], [[] for _ in range(5)]
        self.demux_cache = []
        self.plot_time_cache, self.plot_tension_cache = [[] for _ in range(5)], [[] for _ in range(5)]
        self.loader = DAQLoader()
        self.delta_t_maxes_cache = []
        self.max_time_cache = []
        self.plate_ref_port = DAQPort("5.1")


    @property
    def x_coords(self):
        return np.array([t.x for t in self.thermistances.values()])

    @property
    def y_coords(self):
        return np.array([t.y for t in self.thermistances.values()])

    @property
    def xy_coords(self):
        return self.x_coords, self.y_coords

    def load_calibration_files(self):
        self.calibration_arrays = []
        calibration_folders_path = home_directory / "calibration_files"
        calibration_files = sorted(calibration_folders_path.glob("*.txt"))
        for file in calibration_files:
            self.calibration_arrays.append(pd.read_csv(file, header=0, sep='\t').to_numpy())
        return self.calibration_arrays

    def setup_thermistance_grid(self):
        self.thermistances = {}
        angles_list = []
        out_ports = [DAQPort("5.13"), DAQPort("5.3"), DAQPort("5.11"), DAQPort("5.7"), DAQPort("5.9"), DAQPort("5.5")]
        # in_ports = [DAQPort("1"), DAQPort("2"), DAQPort("3")]
        self.ports = out_ports # + in_ports
        for i in range(6):
            angle = i * np.pi / 3 + np.pi/6
            angles_list.append(angle)
            self.thermistances[out_ports[i]] = Thermistance((self.r_out, angle), out_ports[i], self.calibration_arrays)

        # self.thermistances[DAQPort("1")] = Thermistance((self.r_int, angles_list[5]), DAQPort("1"), self.calibration_arrays)
        # self.thermistances[DAQPort("2")] = Thermistance((self.r_int, angles_list[1]), DAQPort("2"), self.calibration_arrays)
        # self.thermistances[DAQPort("3")] = Thermistance((self.r_int, angles_list[3]), DAQPort("3"), self.calibration_arrays)

        return self.thermistances, self.ports
    
    def device_detected(self):
        system = System.local()
        devices = system.devices
        return True if len(devices) != 0 else False

    def get_plugged_device(self):
        if self.device_detected():
            system = System.local()
            devices = system.devices
            device_name = re.search(r'Device\(name=(.+?)\)', str(devices[0])).group(1)
            return device_name
        else:
            return

    def setup_bits_list(self):
        self.bits_list = []
        for i in range(16):
            bits = [bool(int(b)) for b in format(i, '04b')]
            self.bits_list.append(bits)
        return self.bits_list

    def start_acquisition(self):
        if self.device_detected():
            print("Task Opened")
            device = self.get_plugged_device()
            self.task = nidaqmx.Task()
            self.task.ai_channels.add_ai_voltage_chan(f"{device}/ai0:4", terminal_config=TerminalConfiguration.RSE)
            self.task.timing.cfg_samp_clk_timing(rate=self.sample_rate, sample_mode=AcquisitionType.FINITE,
                                                 samps_per_chan=self.samples_per_read)
            self.reader = AnalogMultiChannelReader(self.task.in_stream)

            self.do_task = nidaqmx.Task()
            self.do_task.do_channels.add_do_chan(
                f"{device}/port0/line0:3",
                line_grouping=LineGrouping.CHAN_PER_LINE
            )

            self.data = np.zeros((5, self.samples_per_read))
            self.start_time = time.time()
            self.i = 0
        else:
            raise RuntimeError("Device not detected")

    def move_bit(self):
        self.i += 1
        if self.i > 15:
            self.i = 0

    def update_cached_data(self, data_to_cache, plot_lines, loading_mode = False):
        if not loading_mode:
            for idx in range(5):
                self.time_cache[idx].append(time.time() - self.start_time)
                self.tension_cache[idx].append(data_to_cache[idx])

                if plot_lines is not None:
                    plot_line = plot_lines[idx]
                    self.plot_tension_cache[idx].append(data_to_cache[idx])
                    self.plot_time_cache[idx].append(time.time() - self.start_time)

                    # Keep only the last 100 points
                    if len(self.plot_time_cache[idx]) > 100:
                        self.plot_time_cache[idx].pop(0)
                        self.plot_tension_cache[idx].pop(0)

                    plot_line.set_xdata(self.plot_time_cache[idx])
                    plot_line.set_ydata(self.plot_tension_cache[idx])
        else:
            self.time_cache, self.tension_cache, self.demux_cache = data_to_cache
            self.plot_tension_cache, self.plot_time_cache = self.tension_cache, self.time_cache
            current_len = self.plot_time_cache.shape[0]
            if plot_lines is not None:
                for idx in range(5):
                    plot_line = plot_lines[idx]
                    if current_len > 100:
                        self.plot_time_cache = self.plot_time_cache[current_len - 100: current_len, :]
                        self.plot_tension_cache = self.plot_tension_cache[current_len - 100: current_len, :]
                    plot_line.set_xdata(self.plot_time_cache[:, idx])
                    plot_line.set_ydata(self.plot_tension_cache[:, idx])

    def fetch_cached_data(self):
        return self.time_cache, self.tension_cache, self.demux_cache

    def fetch_daq_data(self, plot_lines=None):
        self.loading_mode = False
        if self.task.is_task_done():
            self.do_task.write(self.bits_list[self.i])

            if self.do_task.is_task_done():
                self.reader.read_many_sample(self.data, number_of_samples_per_channel=self.samples_per_read)
                averaged_data = np.abs(np.mean(self.data, axis=1))

                self.update_cached_data(averaged_data, plot_lines)

                self.move_bit()
                self.demux_cache.append(self.i)
                if len(self.demux_cache) >= 16:
                    self.update_thermistances_data()
        return plot_lines

    def fetch_simulation_data(self, load_index, plot_lines=None):
        self.loading_mode = True
        data_to_cache = self.loader.load_save_for_ui(load_index)
        self.update_cached_data(data_to_cache, plot_lines, loading_mode=True)
        if len(self.demux_cache) >= 16:
            self.update_thermistances_data()

    def stop_acquisition(self):
        print("Task Closed")
        self.task.close()
        self.do_task.close()

    def reset_data(self):
        self.time_cache, self.tension_cache = [[] for _ in range(5)], [[] for _ in range(5)]
        self.demux_cache = []
        print("Data reset")

    def save_current_data(self, wavelength: float = 900):
        save_folder_path = home_directory / "Saves"
        current_time = str(datetime.datetime.now())
        formatted_name = current_time.replace(" ", "_").replace(":", "_")[:-7]
        save_path = save_folder_path / f"QcWatt_{formatted_name}"
        save_path.mkdir(parents=True, exist_ok=True)
        bits_array = np.array(self.demux_cache)
        time_data_array = np.array(self.time_cache).T
        tension_data_array = np.array(self.tension_cache).T
        save_path_time = save_path / "time.npy"
        save_path_tension = save_path / "tension.npy"
        save_path_bits = save_path / "bits.npy"
        np.save(save_path_time, time_data_array)
        np.save(save_path_tension, tension_data_array)
        np.save(save_path_bits, bits_array)
        return save_path_bits

    def get_port_values(self, port:DAQPort, cached_data: tuple):
        if not self.loading_mode:
            time_list, tension_list, demux_list = cached_data
            bits_array = np.array(demux_list)
            channel_data = np.array([time_list[port.daq_port - 1], tension_list[port.daq_port - 1]])
        else:
            full_time_array, full_tension_array, bits_array = cached_data
            channel_data = np.vstack([full_time_array[:, port.daq_port - 1], full_tension_array[:, port.daq_port - 1]])
        if port.demux_port:
            mask = bits_array == port.demux_port
            channel_data = channel_data[:, mask]  # Apply the mask to the appropriate dimension (columns)
        # print(f"Port: {port}")
        # print(f"Channel Data: {channel_data}")
        return channel_data

    def update_thermistances_data(self):
        cached_data = self.fetch_cached_data()
        for port in self.ports:
            port_data = self.get_port_values(port, cached_data)
            self.thermistances[port].add_data(port_data)

    def get_temperature_values(self):
        plate_ref_thermistance = Thermistance((0, 0), DAQPort("5.1"), self.calibration_arrays)
        plate_ref_value = self.get_port_values(DAQPort("5.1"), self.fetch_cached_data())
        if np.any(plate_ref_value):
            plate_ref_thermistance.add_data(plate_ref_value)
            plate_ref_temp = plate_ref_thermistance.get_temperature()
        else:
            plate_ref_temp = 0
        return [thermistance.get_temperature() - plate_ref_temp for thermistance in self.thermistances.values()]

    def get_laser_params(self):
        rotation_angle = np.radians(0)
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])
        factor = 1.8
        print(self.get_temperature_values())
        try:
            popt, _ = opt.curve_fit(
                gaussian_2d,
                self.xy_coords,
                self.get_temperature_values(),
                p0=self.laser_initial_guesses,
                bounds=([0, -15, -15, 5, 5], [60, 15, 15, 10, 10]),
                maxfev=1000,
            )

            if popt[0] < 0.6:
                popt[1], popt[2] = 0, 0
            else:
                popt[1], popt[2] = np.dot(rotation_matrix, [popt[1] * factor, popt[2] * 1.8])
            self.laser_params = popt
            if abs(popt[1]) < 15 and abs(popt[2]) < 15:
                self.laser_initial_guesses = self.laser_params
            else:
                self.laser_initial_guesses = [10, 0, 0, 8.5, 8.5]
        except RuntimeError:
            print("Couldn't fit data")
            self.laser_params = self.laser_initial_guesses
        return self.laser_params

    def n_glasses(self, lambda_: float):
        ns = []
        for glass in self.glasses:
            a_coeffs, b_coeffs, c_coeffs = (
                glass.n_properties["a"],
                glass.n_properties["b"],
                glass.n_properties["c"],
            )
            a = a_coeffs[0] * lambda_ ** 2 / (lambda_ ** 2 - a_coeffs[1])
            b = b_coeffs[0] * lambda_ ** 2 / (lambda_ ** 2 - b_coeffs[1])
            c = c_coeffs[0] * lambda_ ** 2 / (lambda_ ** 2 - c_coeffs[1])
            ns.append(np.sqrt(1 + a + b + c))
        return ns

    def get_laser_position(self):
        if self.laser_params is None:
            return self.laser_initial_guesses[1], self.laser_initial_guesses[2]
        return self.laser_params[1], self.laser_params[2]

    def estimate_power(self, current_time, delta_max):
        factor = 1/3.2
        factor_2 = 4
        self.delta_t_maxes_cache.append(delta_max)
        self.max_time_cache.append(current_time)
        if len(self.delta_t_maxes_cache) > 20:
            self.delta_t_maxes_cache = self.delta_t_maxes_cache[1:] # Keeping the last 20 only
            self.max_time_cache = self.max_time_cache[1:]
        if len(self.delta_t_maxes_cache) > 1:
            delta_t_array = np.array(self.delta_t_maxes_cache)
            p_mean = np.mean(delta_t_array * factor)
            delta_p = np.diff(delta_t_array * factor)
            delta_t = np.diff(np.array(self.max_time_cache))
            p_est = np.mean(delta_p/delta_t) * factor_2 + p_mean
        else:
            p_est = 0
            print("Insufficient data to estimate power, returning 0W")
        print("Current estimated power:", p_est)
        return np.round(p_est, 2)

    def estimate_absorbance_of_glass(self, temps: list, glass_type: Glass):
        pass

    def estimate_wavelength(self):
        return 976  # For now, will implement actual function later

    def get_wavelength(self):
        if self.manual_wavelength is not None:
            return self.manual_wavelength
        else:
            return self.estimate_wavelength()


if __name__ == "__main__":
    pass