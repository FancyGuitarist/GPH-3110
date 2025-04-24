import numpy as np
import re
from enum import StrEnum
import sys
import time

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


class DAQReader:
    def __init__(self, samples_per_read=10, sample_rate=10_000):
        self.samples_per_read, self.sample_rate = samples_per_read, sample_rate
        self.task, self.reader, self.do_task, self.start_time, self.data = None, None, None, None, None
        self.i = 0
        self.bits_list = self.setup_bits_list()

    def setup_bits_list(self):
        self.bits_list = []
        for i in range(16):
            bits = [bool(int(b)) for b in format(i, '04b')]
            self.bits_list.append(bits)
        return self.bits_list

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

    def fetch_daq_data(self):
        if self.device_detected():
            if self.task.is_task_done():
                self.do_task.write(self.bits_list[self.i])

                if self.do_task.is_task_done():
                    self.reader.read_many_sample(self.data, number_of_samples_per_channel=self.samples_per_read)
                    averaged_data = np.abs(np.mean(self.data, axis=1))
                    current_time = time.time() - self.start_time

                    self.move_bit()
            return current_time, averaged_data, self.i
        else:
            raise RuntimeError("PowerMeter not detected")

    def stop_acquisition(self):
        print("Task Closed")
        self.task.close()
        self.do_task.close()