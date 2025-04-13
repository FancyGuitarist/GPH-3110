import tkinter as tk
import customtkinter as ctk
import numpy as np
import time
import threading
from enum import StrEnum
import matplotlib.colors as mcolors
from packages.powermeter_functions import gaussian_2d, PowerMeter
from PIL import Image, ImageTk
from pathlib import Path
from skimage.draw import disk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ctypes


try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # Enable DPI awareness
except Exception:
    pass


home_directory = Path(__file__).parents[0]
SAMPLE_RATE = 10000
SAMPLES_PER_READ = 100


def setup_grid(self, rows: int, cols: int):
    """
    Sets up the grid layout to place the buttons automatically in the current frame.
    """
    for row in range(rows):
        self.grid_rowconfigure(row, weight=1, uniform="row")
    for col in range(cols):
        self.grid_columnconfigure(col, weight=1, uniform="col")


ctk.CTkFrame.setup_grid = setup_grid


def update_text_box(self, text, center=True):
    """
    Updates the text box with the given text.
    """
    self.configure(state="normal")
    self.delete("0.0", tk.END)
    if center:
        text = " " * 20 + text
    self.insert("0.0", text)
    self.configure(state="disabled")


ctk.CTkTextbox.update_text_box = update_text_box


def random_ui_values():
    """
    Generates random values for the UI.
    """
    power = np.random.randint(900, 1000) / 100
    wavelength = np.random.randint(900, 1000)
    return power, wavelength


class UIColors(StrEnum):
    White = "#F0EFEF"
    LightGray = "#B6B5B5"
    DarkGray = "#3D3F41"
    Black = "#202020"

    @property
    def red_value(self):
        return int(self.value[1:3], 16)

    @property
    def blue_value(self):
        return int(self.value[3:5], 16)

    @property
    def green_value(self):
        return int(self.value[5:], 16)

    @property
    def rgb_value(self):
        return np.array([self.red_value, self.green_value, self.blue_value]).astype(
            np.uint8
        )


class PowerMeterUI(ctk.CTk):
    def __init__(self, *args, **kwargs):
        # Initializing the Tkinter window that will hold the app
        ctk.CTk.__init__(self, *args, **kwargs)
        self.system_width, self.system_height = (
            self.winfo_screenwidth(),
            self.winfo_screenheight() - 70,
        )
        self.screen_dims = (self.system_width, self.system_height)
        self.geometry(f"{self.system_width}x{self.system_height}")
        self.minsize(self.system_width, self.system_height)
        self.maxsize(self.system_width, self.system_height)

        # Variables to be shared between frames
        self.power_meter = PowerMeter()
        self.updating_plot = False
        self.reading_daq = False

        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(-1, 6)  # Adjust as needed
        self.lines = [self.ax.plot([], [])[0] for _ in range(5)]
        self.app_dims = (750, 750)


        # creating a container
        container = ctk.CTkFrame(self)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # initializing frames to an empty dictionary
        self.frames = {}

        # iterating through the different frame layouts
        for F in (MainWindow, DAQReadingsWindow):
            frame = F(container, self)
            # Initializing frames for MainPage and Heat Map.
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Opens app on MainWindow
        self.show_frame(MainWindow)

    # to display the current frame passed as parameter
    def show_frame(self, window):
        """
        Function to display the window passed as parameter.
        """
        frame = self.frames[window]
        frame.tkraise()
        self.geometry(f"{self.app_dims[0]}x{self.app_dims[1]}")
        self.minsize(750, 750)
        self.maxsize(750, 750)
        if frame == self.frames[DAQReadingsWindow]:
            print("Moving to DAQ Readings")
            self.updating_plot = True
            frame.update_plot()
        else:
            self.updating_plot = False

    def get_wavelength(self):
        return self.frames[MainWindow].wavelength_txt_box.get('1.0', tk.END)

    def relx_pos(self, absx_pos):
        return absx_pos / self.app_dims[0]

    def rely_pos(self, absy_pos):
        return absy_pos / self.app_dims[1]


class MainWindow(ctk.CTkFrame):
    """
    Main window of the app, displays power (W), wavelength (nm) and heat map.
    """

    def __init__(self, master, controller):
        super().__init__(master)
        self.master = master
        self.controller = controller
        self.configure(fg_color=UIColors.White)
        self.setup_grid(6, 3)
        self.label_font = ctk.CTkFont(family="Times New Roman", size=20, weight="bold")
        self.text_font = ctk.CTkFont(family="Times New Roman", size=15)
        self.power_meter = self.controller.power_meter
        self.mask_path = home_directory / "ressources" / "Plate.png"
        self.plate_mask_cache, self.circular_mask_cache = None, None
        self.img_tk = None
        self.X, self.Y = np.meshgrid(np.linspace(-1, 1, 400), np.linspace(-1, 1, 350))
        self.activation_count = 0
        self.power_txt_box = ctk.CTkTextbox(
            self,
            width=200,
            height=20,
            corner_radius=10,
            font=self.text_font,
            bg_color=UIColors.White,
            fg_color=UIColors.LightGray,
            text_color=UIColors.Black,
        )
        self.power_txt_box_label = ctk.CTkLabel(
            self, text="Power (W)", font=self.label_font, text_color=UIColors.Black
        )
        self.wavelength_txt_box = ctk.CTkTextbox(
            self,
            width=200,
            height=20,
            corner_radius=10,
            font=self.text_font,
            bg_color=UIColors.White,
            fg_color=UIColors.LightGray,
            text_color=UIColors.Black,
        )
        self.wavelength_txt_box_label = ctk.CTkLabel(
            self,
            text="Wavelength (nm)",
            font=self.label_font,
            text_color=UIColors.Black,
        )
        self.start_acquisition_button = ctk.CTkButton(
            self,
            text="Start Acquisition",
            bg_color=UIColors.White,
            fg_color=UIColors.LightGray,
            text_color=UIColors.Black,
            hover_color=UIColors.DarkGray,
            corner_radius=5,
            border_width=2,
            font=self.label_font,
            width=60,
            height=30,
            command=self.start_acquisition_daq,
        )
        self.stop_acquisition_button = ctk.CTkButton(
            self,
            text="Stop Acquisition",
            bg_color=UIColors.White,
            fg_color=UIColors.LightGray,
            text_color=UIColors.Black,
            hover_color=UIColors.DarkGray,
            corner_radius=5,
            border_width=2,
            font=self.label_font,
            width=60,
            height=30,
            state="disabled",
            command=self.stop_acquisition_daq,
        )

        self.save_data_button = ctk.CTkButton(
            self,
            text="Save Data",
            bg_color=UIColors.White,
            fg_color=UIColors.LightGray,
            text_color=UIColors.Black,
            hover_color=UIColors.DarkGray,
            corner_radius=5,
            border_width=2,
            font=self.label_font,
            width=60,
            height=30,
            state="disabled",
            command=self.power_meter.save_current_data,
        )

        self.reset_data_button = ctk.CTkButton(
            self,
            text="Reset Data",
            bg_color=UIColors.White,
            fg_color=UIColors.LightGray,
            text_color=UIColors.Black,
            hover_color=UIColors.DarkGray,
            corner_radius=5,
            border_width=2,
            font=self.label_font,
            width=60,
            height=30,
            state="disabled",
            command=self.reset_data,
        )

        self.daq_display_button = ctk.CTkButton(
            self,
            text="Daq Display",
            bg_color=UIColors.White,
            fg_color=UIColors.LightGray,
            text_color=UIColors.Black,
            hover_color=UIColors.DarkGray,
            corner_radius=5,
            border_width=2,
            font=self.label_font,
            width=60,
            height=30,
            state="disabled",
            command=lambda: self.controller.show_frame(DAQReadingsWindow),
        )
        self.canvas = ctk.CTkCanvas(
            self, width=400, height=350, bg=UIColors.White, highlightthickness=0
        )
        self.canvas.place(relx=self.controller.relx_pos(213), rely=self.controller.rely_pos(250), anchor=ctk.NW)
        self.image_id = self.canvas.create_image(0, 0, anchor=ctk.NW)

        self.power_txt_box.grid(row=0, column=1, padx=10, pady=10)
        self.power_txt_box_label.place(relx=self.controller.relx_pos(335), rely=self.controller.rely_pos(12), anchor=ctk.NW)
        self.wavelength_txt_box.grid(row=1, column=1, padx=10, pady=10)
        self.wavelength_txt_box_label.place(relx=self.controller.relx_pos(300), rely=self.controller.rely_pos(135), anchor=ctk.NW)
        self.stop_acquisition_button.place(relx=self.controller.relx_pos(395), rely=self.controller.rely_pos(575), anchor=ctk.NW)
        self.start_acquisition_button.place(relx=self.controller.relx_pos(195), rely=self.controller.rely_pos(575), anchor=ctk.NW)
        self.daq_display_button.place(relx=self.controller.relx_pos(75), rely=self.controller.rely_pos(45), anchor=ctk.NW)
        self.save_data_button.place(relx=self.controller.relx_pos(325), rely=self.controller.rely_pos(625), anchor=ctk.NW)
        self.reset_data_button.place(relx=self.controller.relx_pos(322), rely=self.controller.rely_pos(675), anchor=ctk.NW)
        threading.Thread(target=self.update_values).start()
        threading.Thread(target=self.update_gradient).start()

    def update_values(self, random=True):
        """
        Updates the power and wavelength values in the UI at a frequency of 30 Hz.
        """
        interval = 1 / 15
        while True:
            if random:
                power, wavelength = random_ui_values()
            else:
                """ Will read values from DAQ in future update """
                power, wavelength = 0, 0
            self.power_txt_box.update_text_box(f"{power}")
            self.wavelength_txt_box.update_text_box(f"{wavelength}")
            time.sleep(interval)

    def get_plate_mask_array(self):
        x_dim = 240
        y_dim = round(x_dim * 1.105)
        pad_y = 350 - y_dim
        pad_x = (400 - x_dim) // 2
        if self.plate_mask_cache is None:
            mask_img = Image.open(self.mask_path)
            mask_img = mask_img.resize((x_dim, y_dim))
            mask_array = np.array(mask_img.convert("RGB"))
            mask_array = np.pad(
                mask_array,
                ((pad_y // 3 * 2, pad_y // 3 + 1), (pad_x, pad_x), (0, 0)),
                mode="constant",
            )
            self.plate_mask_cache = np.invert(mask_array.astype(np.bool))
        return self.plate_mask_cache

    def get_circular_mask_array(self):
        if self.circular_mask_cache is None:
            self.circular_mask_cache = np.zeros((350, 400, 3), dtype=np.uint8)
            rr, cc = disk((185, 200), 90)
            self.circular_mask_cache[rr, cc, :] = 1
        return self.circular_mask_cache

    def apply_masks_to_gradient(self, img_array):
        circular_mask_array = self.get_circular_mask_array()
        background = (
            np.invert(circular_mask_array.astype(np.bool)) * UIColors.White.rgb_value
        )
        masked_array = (
            img_array * circular_mask_array + background
        ) * self.get_plate_mask_array()
        return masked_array

    def update_gradient(self):
        params = self.power_meter.get_laser_params()
        Z = gaussian_2d((self.X, self.Y), *params)

        # Normalize the data to fit a colormap
        norm = mcolors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
        colormap = plt.get_cmap("coolwarm")

        # Convert the Gaussian values to RGB colors
        img_array = (colormap(norm(Z))[:, :, :3] * 255).astype(np.uint8)
        img_array = self.apply_masks_to_gradient(img_array)

        # Convert to PIL Image
        img = Image.fromarray(img_array)

        # Convert to Tkinter-compatible format
        self.img_tk = ImageTk.PhotoImage(img)

        # Update the label's image
        self.canvas.itemconfig(self.image_id, image=self.img_tk)

        self.draw_overlay_shapes(norm, colormap)

        # Schedule next update (simulate real-time data)
        self.canvas.after(1000, self.update_gradient)  # Update every second

    def get_ui_thermistance_positions(self):
        positions = np.array(self.power_meter.xy_coords)
        positions[0, :] -= np.min(positions[0, :])
        positions[1, :] -= np.min(positions[1, :])
        positions /= np.max(positions)
        positions *= 200
        return positions + 100

    def draw_overlay_shapes(self, norm, colormap):
        """Draw circles or other shapes on top of the heatmap"""
        self.canvas.delete("overlay")  # Remove previous shapes
        temps = self.power_meter.get_temperature_values()
        positions = self.get_ui_thermistance_positions()

        for index in range(len(temps)):
            x, y = positions[:, index]
            current_color = colormap(norm(temps[index]))
            current_color_hex = mcolors.rgb2hex(current_color)
            self.canvas.create_oval(
                x - 5,
                y - 5,
                x + 5,
                y + 5,
                fill=str(current_color_hex),
                outline="Black",
                width=2,
                tags="overlay",
            )

    def start_acquisition_daq(self):
        try:
            self.power_meter.start_acquisition()
        except RuntimeError:
            print("No Device Detected")
            return
        if self.activation_count == 0:
            self.read_daq_data()
            self.activation_count += 1
        self.start_acquisition_button.configure(state="disabled")
        self.stop_acquisition_button.configure(state="normal")
        self.daq_display_button.configure(state="normal")
        self.save_data_button.configure(state="disabled")
        self.reset_data_button.configure(state="disabled")
        self.controller.reading_daq = True

    def stop_acquisition_daq(self):
        self.controller.updating_plot = False
        self.controller.reading_daq = False
        self.after(11)
        self.power_meter.stop_acquisition()
        self.start_acquisition_button.configure(state="normal")
        self.stop_acquisition_button.configure(state="disabled")
        self.daq_display_button.configure(state="disabled")
        self.save_data_button.configure(state="normal")
        self.reset_data_button.configure(state="normal")

    def reset_data(self):
        self.power_meter.reset_data()
        self.stop_acquisition_button.configure(state="disabled")
        self.start_acquisition_button.configure(state="normal")
        self.save_data_button.configure(state="disabled")

    def read_daq_data(self):
        if self.controller.reading_daq:
            self.controller.lines = self.power_meter.fetch_daq_data(self.controller.lines)
            if self.controller.updating_plot:
                self.controller.frames[DAQReadingsWindow].update_plot()
        self.after(10, self.read_daq_data)


class DAQReadingsWindow(ctk.CTkFrame):
    """
    Window for viewing the DAQ readings in real time.
    """

    def __init__(self, master, controller):
        super().__init__(master)
        self.master = master
        self.controller = controller
        self.power_meter = self.controller.power_meter
        self.label_font = ctk.CTkFont(family="Times New Roman", size=20, weight="bold")

        self.show_main_window_button = ctk.CTkButton(
            self,
            text="Powermeter View",
            bg_color=UIColors.White,
            fg_color=UIColors.LightGray,
            text_color=UIColors.Black,
            hover_color=UIColors.DarkGray,
            corner_radius=5,
            border_width=2,
            font=self.label_font,
            width=60,
            height=30,
            command=lambda: self.controller.show_frame(MainWindow),
        )

        # Create the plot
        self.fig, self.ax = self.controller.fig, self.controller.ax
        self.lines = self.controller.lines

        # Embed the plot into the Tkinter Frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.show_main_window_button.pack()


    def update_plot(self):
        """
        Function to update the DAQ readings and refresh the plot.
        """
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()


if __name__ == "__main__":
    app = PowerMeterUI()
    app.title("Power Meter Interface v0.2.2")
    app.mainloop()
