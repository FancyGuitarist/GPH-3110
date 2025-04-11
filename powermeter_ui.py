import tkinter as tk
import customtkinter as ctk
import numpy as np
import time
import datetime
import threading
from enum import StrEnum
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from packages.functions import gaussian_2d, PowerMeter
from PIL import Image, ImageTk
from pathlib import Path
from skimage.draw import disk
import pandas as pd
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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
        self.geometry("750x750")
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
        self.acquisition_button = ctk.CTkButton(
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
        self.canvas.place(x=268, y=320)
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW)

        self.power_txt_box.grid(row=0, column=1, padx=10, pady=10)
        self.power_txt_box_label.place(x=335, y=12)
        self.wavelength_txt_box.grid(row=1, column=1, padx=10, pady=10)
        self.wavelength_txt_box_label.place(x=300, y=135)
        self.stop_acquisition_button.place(x=395, y=575)
        self.acquisition_button.place(x=195, y=575)
        self.daq_display_button.place(x=75, y=45)
        self.save_data_button.place(x=325, y=625)
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
        colormap = cm.get_cmap("coolwarm")

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
        self.power_meter.start_acquisition()
        self.acquisition_button.configure(state="disabled")
        self.stop_acquisition_button.configure(state="normal")
        self.daq_display_button.configure(state="normal")
        self.save_data_button.configure(state="disabled")

    def stop_acquisition_daq(self):
        self.controller.updating_plot = False
        self.after(11)
        self.power_meter.stop_acquisition()
        self.acquisition_button.configure(state="normal")
        self.stop_acquisition_button.configure(state="disabled")
        self.daq_display_button.configure(state="disabled")
        self.save_data_button.configure(state="normal")


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

        self.save_stored_data_button = ctk.CTkButton(
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
            command=lambda: self.power_meter.save_current_data(),
        )

        # Create the plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(-1, 6)  # Adjust as needed
        self.lines = [self.ax.plot([], [])[0] for _ in range(5)]
        self.x_data = [[] for _ in range(5)]
        self.y_data = [[] for _ in range(5)]
        self.x_data_store = [[] for _ in range(5)]  # Will be moved to PowerMeter Class
        self.y_data_store = [[] for _ in range(5)]  # Will be moved to PowerMeter Class
        self.demux_bits = []

        # Embed the plot into the Tkinter Frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.show_main_window_button.pack()
        self.save_stored_data_button.pack()


    def update_plot(self):
        """
        Function to update the DAQ readings and refresh the plot.
        """
        self.lines = self.power_meter.fetch_daq_data(self.lines)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

        # Schedule next update
        if self.controller.updating_plot:
            self.after(10, self.update_plot)  # Update every 10ms

    def save_current_data(self):
        save_folder_path = home_directory / "Saves"
        save_path = save_folder_path / f"QcWatt_{datetime.datetime.now().date()}_{int(self.controller.get_wavelength())}"
        save_path.mkdir(parents=True, exist_ok=True)
        bits_array = np.array(self.demux_bits)
        x_data_array = np.array(self.x_data_store).T
        y_data_array = np.array(self.y_data_store).T
        save_path_time = save_path / "time.npy"
        save_path_tension = save_path / "tension.npy"
        save_path_bits = save_path / "bits.npy"
        np.save(save_path_time, x_data_array)
        np.save(save_path_tension, y_data_array)
        np.save(save_path_bits, bits_array)


    def close(self):
        """
        Clean up the DAQ tasks when closing the window.
        """
        self.task.close()
        self.do_task.close()


if __name__ == "__main__":
    app = PowerMeterUI()
    app.title("Power Meter Interface v0.2.2")
    # app.after(100, lambda: app.focus_force())
    # if platform.system() == "Windows":
    #     print("Ah shit, it's Windows")
    #     app.attributes("-alpha", 1.0)
    #     app.attributes("-transparentcolor", "")
    #     hwnd = app.winfo_id()
    #     ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0, 255, 2)
    app.mainloop()
