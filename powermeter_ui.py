import tkinter as tk
import customtkinter as ctk
import nidaqmx
import numpy as np
import time
import threading
from queue import Queue
from enum import StrEnum
import matplotlib.colors as mcolors
from packages.powermeter_functions import gaussian_2d, PowerMeter
from packages.daq_reader import DAQReader
from packages.daq_data_loader import  DAQLoader
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
    Green = "#34EB71"
    LightGreen = "#98ebb4"
    Red = "#F70717"
    LightRed = "#f2555f"
    Blue = "#6e74eb"
    LightBlue = "#a0a3eb"
    Orange = "#E09525"

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

class UIMode(StrEnum):
    Automatic = "automatic"
    Manual = "manual"

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
        self.exam_mode = True
        self.power_meter = PowerMeter()
        self.daq_reader = DAQReader()
        self.daq_loader = DAQLoader()
        self.reading_daq = False
        self.reading_save = False

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
        # App setup methods and items
        self.master = master
        self.controller = controller
        self.configure(fg_color=UIColors.White)
        self.setup_grid(6, 3)
        self.label_font = ctk.CTkFont(family="Times New Roman", size=20, weight="bold")
        self.toggle_font = ctk.CTkFont(family="Times New Roman", size=18, weight="bold")
        self.text_font = ctk.CTkFont(family="Times New Roman", size=15)
        self.power_meter = self.controller.power_meter
        self.daq_reader = self.controller.daq_reader
        self.daq_loader = self.controller.daq_loader
        self.data_queue = Queue()
        self.daq_thread = None
        self.loader_thread = None
        self.resources_path = home_directory / "resources"
        self.mask_path = self.resources_path/ "Plate.png"
        self.on_img = ctk.CTkImage(Image.open(self.resources_path / "on-toggle.png"), size=(50, 50))
        self.off_img = ctk.CTkImage(Image.open(self.resources_path / "off-toggle.png"), size=(50, 50))
        # App variables
        self.toggle_state = ctk.StringVar(value=UIMode.Automatic)
        self.manual_wavelength = ctk.StringVar()
        self.plate_mask_cache, self.circular_mask_cache = None, None
        self.img_tk = None
        self.canvas_background = (np.ones((350, 400, 3)) * UIColors.LightGray.rgb_value).astype(np.uint8)
        self.X, self.Y = np.meshgrid(np.linspace(-13.97, 13.97, 400), np.linspace(-13.97, 13.97, 350))
        self.estimated_power = 0
        self.activation_count = 0
        self.elapsed_time = 0
        self.iteration = 0
        self.current_acquisition_time = 0
        self.updating_gradient = False
        self.status_placement = (18, 162)
        self.status_border_height = 60
        self.status_border_width = 300
        self.power_txt_box = ctk.CTkTextbox(
            self,
            width=200,
            height=20,
            corner_radius=10,
            font=self.label_font,
            bg_color=UIColors.White,
            fg_color=UIColors.LightGray,
            text_color=UIColors.Black,
        )
        self.power_txt_box_label = ctk.CTkLabel(
            self, text="Puissance (W)", font=self.label_font, text_color=UIColors.Black
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
            text="Longueur d'onde (nm)",
            font=self.label_font,
            text_color=UIColors.Black,
        )
        self.start_acquisition_button = ctk.CTkButton(
            self,
            text="Lancer Acquisition",
            bg_color=UIColors.White,
            fg_color=UIColors.Green,
            text_color=UIColors.Black,
            hover_color=UIColors.LightGreen,
            corner_radius=5,
            border_width=2,
            font=self.label_font,
            width=60,
            height=30,
            command=self.start_acquisition_daq,
        )
        self.stop_acquisition_button = ctk.CTkButton(
            self,
            text="Arrêter Acquisition",
            bg_color=UIColors.White,
            fg_color=UIColors.Red,
            text_color=UIColors.Black,
            hover_color=UIColors.LightRed,
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
            text="Sauvegarder Données",
            bg_color=UIColors.White,
            fg_color=UIColors.Blue,
            text_color=UIColors.Black,
            hover_color=UIColors.DarkGray,
            corner_radius=5,
            border_width=2,
            font=self.label_font,
            width=60,
            height=30,
            state="disabled",
            command=self.save_data,
        )

        self.reset_data_button = ctk.CTkButton(
            self,
            text="Réinitialiser Données",
            bg_color=UIColors.White,
            fg_color=UIColors.Black,
            text_color=UIColors.White,
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

        self.save_selector = ctk.CTkComboBox(
            self,
            values = self.daq_loader.combobox_options + ["None"],
            font=self.text_font,
            dropdown_font=self.text_font,
            state="readonly",
            width=175,
        )
        self.save_selector.set("None")

        self.save_selector_label = ctk.CTkLabel(
            self,
            text="Fichier à charger",
            font=self.label_font,
            text_color=UIColors.Black,
        )

        self.load_button = ctk.CTkButton(
            self,
            text="Charger Sauvegarde",
            bg_color=UIColors.White,
            fg_color=UIColors.LightGray,
            text_color=UIColors.Black,
            hover_color=UIColors.DarkGray,
            corner_radius=5,
            border_width=2,
            font=self.label_font,
            width=60,
            height=30,
            command=self.start_loading_data,
        )

        self.pause_loading_button = ctk.CTkButton(
            self,
            text="Arrêter Chargement",
            bg_color=UIColors.White,
            fg_color=UIColors.LightGray,
            text_color=UIColors.Black,
            hover_color=UIColors.DarkGray,
            corner_radius=5,
            border_width=2,
            font=self.label_font,
            width=60,
            height=30,
            command=self.pause_loading_data,
        )

        self.position_label = ctk.CTkLabel(
            self,
            text= "Position actuelle:" + "\n" + "x: 0.00 mm" + "\n" + "y: 0.00 mm",
            width=60,
            height=30,
            font=self.label_font,
        )

        self.time_label = ctk.CTkLabel(
            self,
            text="Durée de l'enregistrement: " + "0.00 sec",
            width=60,
            height=30,
            font=self.label_font,
        )

        self.status_border_frame = tk.Frame(
            self,
            bg=UIColors.Orange,
            width=self.status_border_width,
            height=self.status_border_height,
        )
        self.status_border_frame.pack_propagate(False)
        self.status_inner_frame = tk.Frame(
            self.status_border_frame,
            bg=UIColors.Black,
            width=self.status_border_width - 10,
            height=self.status_border_height - 10,
        )
        self.status_label = tk.Label(
            self.status_inner_frame,
            text="Brancher le Puissance-Mètre pour débuter l'acquisition",
            bg=UIColors.Black,
            fg=UIColors.White,
            justify="center",
            wraplength=self.status_border_width - 10,
        )

        # Toggle setup, not used for the time being
        self.toggle_button = ctk.CTkButton(
            self,
            image=self.off_img,
            text="",
            fg_color="transparent",
            height=50,
            width=50,
            anchor="center",
            hover_color=UIColors.White,
            command=self.update_toggle_state,
        )

        self.toggle_button_label = ctk.CTkLabel(
            self,
            text="Acquisition longueur d'onde",
            font=self.toggle_font,
            text_color=UIColors.Black,
        )

        self.manual_label = ctk.CTkLabel(
            self,
            text="Manuelle",
            font=self.toggle_font,
            text_color=UIColors.Black,
        )

        self.automatic_label = ctk.CTkLabel(
            self,
            text="Automatique",
            font=self.toggle_font,
            text_color=UIColors.Black,
        )

        self.wavelength_entry = ctk.CTkEntry(
            self,
            width=90,
            font=self.label_font,
            bg_color=UIColors.White,
            fg_color=UIColors.White,
            text_color=UIColors.Black,
            textvariable=self.manual_wavelength,
            validate="all",
            validatecommand=(self.register(self.restrict_entry), "%P"),
            justify="center",
            state="disabled"
        )

        self.wavelength_button = ctk.CTkButton(
            self,
            text="Confirmer",
            fg_color=UIColors.LightGray,
            bg_color=UIColors.White,
            text_color=UIColors.Black,
            font=self.label_font,
            height=30,
            width=50,
            anchor="center",
            state="disabled",
            hover_color=UIColors.DarkGray,
            command=self.submit_wavelength,
        )

        self.canvas = ctk.CTkCanvas(
            self, width=400, height=350, bg=UIColors.White, highlightthickness=0
        )
        self.canvas.place(relx=self.controller.relx_pos(213), rely=self.controller.rely_pos(250), anchor=ctk.NW)
        self.image_id = self.canvas.create_image(0, 0, anchor=ctk.NW)

        self.power_txt_box.grid(row=0, column=1, padx=10, pady=10)
        self.power_txt_box_label.place(relx=self.controller.relx_pos(315), rely=self.controller.rely_pos(12), anchor=ctk.NW)
        self.wavelength_txt_box.grid(row=1, column=1, padx=10, pady=10)
        self.wavelength_txt_box_label.place(relx=self.controller.relx_pos(280), rely=self.controller.rely_pos(135), anchor=ctk.NW)
        self.stop_acquisition_button.place(relx=self.controller.relx_pos(390), rely=self.controller.rely_pos(575), anchor=ctk.NW)
        self.start_acquisition_button.place(relx=self.controller.relx_pos(190), rely=self.controller.rely_pos(575), anchor=ctk.NW)
        self.save_data_button.place(relx=self.controller.relx_pos(280), rely=self.controller.rely_pos(625), anchor=ctk.NW)
        self.reset_data_button.place(relx=self.controller.relx_pos(280), rely=self.controller.rely_pos(675), anchor=ctk.NW)
        if not self.controller.exam_mode:
            selector_pos = (530, 50)
            self.save_selector.place(relx=self.controller.relx_pos(selector_pos[0]), rely=self.controller.rely_pos(selector_pos[1]), anchor=ctk.NW)
            self.save_selector_label.place(relx=self.controller.relx_pos(selector_pos[0] + 10), rely=self.controller.rely_pos(selector_pos[1] - 30), anchor=ctk.NW)
            self.load_button.place(relx=self.controller.relx_pos(selector_pos[0] - 5), rely=self.controller.rely_pos(selector_pos[1] + 50), anchor=ctk.NW)
            self.pause_loading_button.place(relx=self.controller.relx_pos(selector_pos[0] - 5), rely=self.controller.rely_pos(selector_pos[1] + 100), anchor=ctk.NW)
        self.position_label.place(relx=self.controller.relx_pos(540), rely=self.controller.rely_pos(350), anchor=ctk.NW)
        self.time_label.place(relx=self.controller.relx_pos(230), rely=self.controller.rely_pos(525), anchor=ctk.NW)
        self.status_border_frame.place(relx=self.controller.relx_pos(self.status_placement[0]), rely=self.controller.rely_pos(self.status_placement[1]) , anchor=ctk.NW)
        self.status_inner_frame.place(relx=0.5, rely=0.5, anchor="center")
        self.status_label.place(relx=0.5, rely=0.5, anchor="center")
        self.wavelength_entry.place(relx=self.controller.relx_pos(95), rely=self.controller.rely_pos(85), anchor=ctk.NW)
        self.wavelength_button.place(relx=self.controller.relx_pos(88), rely=self.controller.rely_pos(123), anchor=ctk.NW)
        # Rest of the Toggle setup
        toggle_pos = (110, 35)
        self.toggle_button.place(relx=self.controller.relx_pos(toggle_pos[0]), rely=self.controller.rely_pos(toggle_pos[1]), anchor=ctk.NW)
        self.toggle_button_label.place(relx=self.controller.relx_pos(toggle_pos[0] - 80), rely=self.controller.rely_pos(toggle_pos[1] - 20), anchor=ctk.NW)
        self.automatic_label.place(relx=self.controller.relx_pos(toggle_pos[0] - 95),
                                       rely=self.controller.rely_pos(toggle_pos[1] + 14), anchor=ctk.NW)
        self.manual_label.place(relx=self.controller.relx_pos(toggle_pos[0] + 68),
                                 rely=self.controller.rely_pos(toggle_pos[1] + 14), anchor=ctk.NW)
        gradient_result = self.calculate_gradient()
        self.update_gradient(gradient_result)
        self.setup_canvas_image()
        threading.Thread(target=self.check_if_daq_connected).start()
        self.after(10, self.update_from_queue)

    def check_if_daq_connected(self):
        if self.daq_reader.device_detected():
            self.update_status_txt_box("Prêt à lancer l'acquisition")
            self.start_acquisition_button.configure(state="normal")
        else:
            self.update_status_txt_box("Brancher le Puissance-Mètre pour débuter l'acquisition")
            if self.start_acquisition_button.cget("state") == "normal":
                self.start_acquisition_button.configure(state="disabled")
            self.after(50, self.check_if_daq_connected)

    def restrict_entry(self, P):
        """
        Function to block any input that is not a digit or more than two digits.
        Used for the overlap text box to restrict the input between 0 and 99%.
        """
        if len(P) > 4:
            return False
        if str.isdigit(P) or P == "":
            return True
        else:
            return False

    def submit_wavelength(self):
        wavelength_txt = self.manual_wavelength.get()
        if wavelength_txt == "":
            self.update_status_txt_box("Entrer une valeur de longueur d'onde")
        else:
            self.power_meter.manual_wavelength = int(wavelength_txt)

    def update_power_and_wavelength(self, power, wavelength):
        """
        Updates the power and wavelength values in the UI at a frequency of 30 Hz.
        """
        if self.controller.reading_daq:
            print("Current Power and Wavelength: ", power, wavelength)
            self.power_txt_box.update_text_box(f"{power}")
            self.wavelength_txt_box.update_text_box(f"{wavelength}")

    def get_plate_mask_array(self):
        if self.plate_mask_cache is None:
            x_dim = 340  # Define width
            y_dim = 316  # Define height
            pad_y = (350 - y_dim) // 2  # Center vertically by evenly distributing padding
            pad_x = (400 - x_dim) // 2  # Center horizontally by evenly distributing padding
            mask_img = Image.open(self.mask_path).rotate(30, expand=1)  # Rotate as needed
            mask_img = mask_img.resize((x_dim, y_dim))  # Resize the mask
            mask_array = np.array(mask_img.convert("RGB"))
            mask_array = np.pad(
                mask_array,
                ((pad_y + 4, pad_y - 4), (pad_x, pad_x), (0, 0)),  # Even padding
                mode="constant",
            )
            self.plate_mask_cache = np.invert(mask_array.astype(bool))  # Adjust for logical inversion

        return self.plate_mask_cache

    def get_circular_mask_array(self):
        if self.circular_mask_cache is None:
            circular_mask = np.zeros((350, 400, 3), dtype=np.uint8)
            rr, cc = disk((180, 200), 90)
            circular_mask[rr, cc, :] = 1
            self.circular_mask_cache = circular_mask, np.invert(circular_mask.astype(np.bool))
        return self.circular_mask_cache

    def apply_masks_to_gradient(self, img_array):
        circular_mask_array, inverse_circular_mask = self.get_circular_mask_array()
        background = inverse_circular_mask * UIColors.White.rgb_value
        masked_array = (img_array * circular_mask_array + background) * self.get_plate_mask_array()
        return masked_array

    def update_position_and_time_ui(self, x0, y0):
        if self.controller.reading_daq:
            position_text = "Position actuelle:" + "\n" + f"x: {x0:.2f} mm" + "\n" + f"y: {y0:.2f} mm"
            if self.daq_reader.start_time is None:
                self.current_acquisition_time = 0
            else:
                self.current_acquisition_time = time.time() - self.daq_reader.start_time + self.elapsed_time
            time_text = "Durée de l'enregistrement: " + f"{self.current_acquisition_time:.2f} sec"
            self.position_label.configure(text=position_text)
            self.time_label.configure(text=time_text)

    def setup_canvas_image(self):
        img_array = self.apply_masks_to_gradient(self.canvas_background)
        img = Image.fromarray(img_array)
        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self.image_id, image=self.img_tk)
    
    def calculate_gradient(self, daq_data=None):
        if daq_data is not None:
            daq_data = self.power_meter.update_data(daq_data)
            params = self.power_meter.get_laser_params(daq_data)
            self.estimated_power = self.power_meter.estimate_power(time.time() - self.daq_reader.start_time, params[0])
            temps = self.power_meter.get_temperature_values(daq_data)
        else:
            params = self.power_meter.laser_initial_guesses
            self.estimated_power = 0
            temps = [0 for _ in range(len(self.power_meter.ports))]
        self.update_position_and_time_ui(params[1], params[2])
        self.power_meter.update_save_cache(self.estimated_power,
                                           self.power_meter.get_wavelength(),
                                           (params[1], params[2]),
                                           self.iteration,
                                           self.daq_reader.start_time)
        self.iteration += 1
        Z = np.flip(gaussian_2d((self.X, self.Y), *params), axis=0)

        # Normalize the data to fit a colormap
        norm = mcolors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
        colormap = plt.get_cmap("hot")

        # Convert to Tkinter-compatible format
        return norm, colormap, (params[1], params[2]), temps, self.estimated_power

    def update_gradient(self, gradient_result):
        norm, colormap, current_pos, temps, self.estimated_power = gradient_result
        self.update_power_and_wavelength(self.estimated_power, self.power_meter.get_wavelength())

        self.draw_overlay_shapes(norm, colormap, current_pos, temps)

    def ui_updates_async(self, daq_data):
        def worker():
            gradient_result = self.calculate_gradient(daq_data)
            self.after(0, lambda: self.update_gradient(gradient_result))
        threading.Thread(target=worker, daemon=True).start()

    def get_ui_thermistor_positions(self):
        positions = np.array(self.power_meter.xy_coords)
        r_out = self.power_meter.r_out
        positions[0, :] = ((positions[0, :] + r_out) / (r_out * 2)) * 200 + 100
        positions[1, :] = (np.abs(positions[1, :] - r_out) / (r_out * 2)) * 175 + 92.5
        return positions

    def convert_radial_to_ui_pos(self, pos):
        x, y = pos
        r_out = self.power_meter.r_out
        x = ((x + r_out) / (r_out * 2)) * 200 + 100
        y = (abs(y - r_out) / (r_out * 2)) * 175 + 92.5
        return x, y

    def draw_overlay_shapes(self, norm, colormap, current_pos, temps):
        """Draw circles or other shapes on top of the heatmap"""
        if not hasattr(self, "overlay_shapes_initialized"):
            # Initialize once
            self.first_oval = self.canvas.create_oval(
                0, 0, 0, 0, fill="Red", outline="Black", width=2, tags="overlay"
            )
            self.other_ovals = []
            positions = self.get_ui_thermistor_positions()
            for index in range(len(positions[0])):
                x, y = positions[:, index]
                oval = self.canvas.create_oval(
                    x - 5, y - 5, x + 5, y + 5,
                    fill="Black", outline="Black", width=2, tags="overlay"
                )
                self.other_ovals.append(oval)
            self.overlay_shapes_initialized = True

        # Update the position and color of the first oval
        x, y = self.convert_radial_to_ui_pos(current_pos)
        self.canvas.coords(self.first_oval, x - 10, y - 10, x + 10, y + 10)

        # Update colors of the other ovals
        for index, oval in enumerate(self.other_ovals):
            current_color = colormap(norm(temps[index]))
            current_color_hex = mcolors.rgb2hex(current_color)
            self.canvas.itemconfig(oval, fill=str(current_color_hex))

    # Toggle command
    def update_toggle_state(self):
        match self.toggle_state.get():
            case UIMode.Automatic:
                self.toggle_state.set(UIMode.Manual)
                self.toggle_button.configure(image=self.on_img)
                self.update_status_txt_box("Entrée manuelle de la longueur d'onde, entrer une valeur pour calibrer la puissance")
                self.wavelength_entry.configure(state="normal")
                self.wavelength_button.configure(state="normal")

            case UIMode.Manual:
                self.toggle_state.set(UIMode.Automatic)
                self.wavelength_entry.delete(0, "end")
                self.power_meter.manual_wavelength = None
                self.toggle_button.configure(image=self.off_img)
                self.update_status_txt_box("Estimation automatique de la longueur d'onde, prêt à lancer l'acquisition")
                self.wavelength_entry.configure(state="disabled")
                self.wavelength_button.configure(state="disabled")

    def update_status_txt_box(self, text):
        self.status_label.configure(text=text, justify="center")

    def start_acquisition_daq(self):
        if self.daq_reader.device_detected():
            self.daq_reader.start_acquisition()
        else:
            threading.Thread(target=self.check_if_daq_connected).start()
            return
        self.daq_thread = threading.Thread(target=self.read_daq_data_loop, daemon=True)
        self.controller.reading_daq = True
        self.daq_thread.start()
        self.update_status_txt_box("Acquisition en cours")
        self.start_acquisition_button.configure(state="disabled")
        self.stop_acquisition_button.configure(state="normal")
        self.daq_display_button.configure(state="normal")
        self.save_data_button.configure(state="disabled")
        self.reset_data_button.configure(state="disabled")


    def stop_acquisition_daq(self):
        self.controller.reading_daq = False
        self.updating_gradient = False
        self.after(11)
        self.daq_thread.join()
        self.elapsed_time = self.current_acquisition_time
        self.update_status_txt_box("Acquisition mise sur pause")
        self.daq_reader.stop_acquisition()
        self.start_acquisition_button.configure(state="normal")
        self.stop_acquisition_button.configure(state="disabled")
        self.daq_display_button.configure(state="disabled")
        self.save_data_button.configure(state="normal")
        self.reset_data_button.configure(state="normal")

    def reset_data(self):
        self.power_meter.reset_data()
        self.daq_reader.start_time = None
        self.current_acquisition_time = 0
        self.elapsed_time = 0
        self.controller.reading_daq = True
        self.update_position_and_time_ui(0, 0)
        self.controller.reading_daq = False
        self.update_status_txt_box("Session d'enregistrement réinitialisée")
        self.stop_acquisition_button.configure(state="disabled")
        self.reset_data_button.configure(state="disabled")
        self.start_acquisition_button.configure(state="normal")
        self.save_data_button.configure(state="disabled")

    def save_data(self):
        save_path = self.power_meter.save_current_data()
        while not save_path.exists():
            print("file isn't saved yet")
            self.after(10)
        self.update_status_txt_box("Données actuelles enregistrées")
        self.daq_loader.get_combobox_options()
        self.save_selector.configure(values=self.daq_loader.combobox_options + ["None"])

    def read_daq_data_loop(self):
        while self.controller.reading_daq:
            try:
                daq_data = self.daq_reader.fetch_daq_data()
                self.data_queue.put(daq_data)
            except Exception:
                self.update_status_txt_box("Puissance-Mètre débranché, veuillez le rebrancher pour repartir l'acquisition")
                self.stop_acquisition_daq()
                threading.Thread(target=self.check_if_daq_connected).start()
        time.sleep(0.001)

    def load_data_loop(self):
        while self.controller.reading_save:
            try:
                daq_data = self.daq_loader.load_save_for_ui()
                self.data_queue.put(daq_data)
            except Exception as e:
                print(f"Error reading DAQ: {e}")
        time.sleep(0.001)

    def update_from_queue(self):
        try:
            while not self.data_queue.empty():
                daq_data = self.data_queue.get_nowait()
                self.ui_updates_async(daq_data)
        except Exception as e:
            print(f"Error updating UI: {e}")
        finally:
            self.after(10, self.update_from_queue)

    def start_loading_data(self):
        self.controller.reading_save = True
        self.daq_loader.load_index = 0
        save_to_load = self.save_selector.get()
        save_index = self.daq_loader.find_combobox_index(save_to_load)
        self.daq_loader.set_save_index(save_index)
        self.loader_thread = threading.Thread(target=self.load_data_loop, daemon=True)
        self.loader_thread.start()
        self.updating_gradient = True
        self.load_button.configure(state="disabled")
        self.pause_loading_button.configure(state="normal")

    def pause_loading_data(self):
        self.controller.reading_save = False
        self.updating_gradient = False
        self.load_button.configure(state="normal")
        self.pause_loading_button.configure(state="disabled")


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
