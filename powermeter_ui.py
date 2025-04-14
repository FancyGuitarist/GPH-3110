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
    Green = "#34EB71"
    LightGreen = "#98ebb4"
    Red = "#F70717"
    LightRed = "#f2555f"
    Blue = "#6e74eb"
    LightBlue = "#a0a3eb"

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
    Acquisition = "acquisition"
    Loading = "loading"
    Neutral = "neutral"

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
        self.toggle_font = ctk.CTkFont(family="Times New Roman", size=18, weight="bold")
        self.text_font = ctk.CTkFont(family="Times New Roman", size=15)
        self.power_meter = self.controller.power_meter
        self.resources_path = home_directory / "resources"
        self.mask_path = self.resources_path/ "Plate.png"
        self.on_img = ctk.CTkImage(Image.open(self.resources_path / "on-toggle.png"), size=(50, 50))
        self.off_img = ctk.CTkImage(Image.open(self.resources_path / "off-toggle.png"), size=(50, 50))
        self.toggle_state = ctk.StringVar(value=UIMode.Neutral)
        self.plate_mask_cache, self.circular_mask_cache = None, None
        self.img_tk = None
        self.X, self.Y = np.meshgrid(np.linspace(-1, 1, 400), np.linspace(-1, 1, 350))
        self.activation_count = 0
        self.elapsed_time = 0
        self.current_acquisition_time = 0
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
            text="Stop Acquisition",
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
            text="Save Data",
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
            command=self.power_meter.save_current_data,
        )

        self.reset_data_button = ctk.CTkButton(
            self,
            text="Reset Data",
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
            values = self.power_meter.loader.combobox_options + ["None"],
            font=self.text_font,
            dropdown_font=self.text_font,
            state="readonly",
            width=175,
        )
        self.save_selector.set("None")

        self.save_selector_label = ctk.CTkLabel(
            self,
            text="File to Load",
            font=self.label_font,
            text_color=UIColors.Black,
        )

        self.load_button = ctk.CTkButton(
            self,
            text="Load Save",
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
            text="Pause Loading",
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
            text= "Current position:" + "\n" + "x: 0.00 mm" + "\n" + "y: 0.00 mm",
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

        # Toggle setup, not used for the time being
        # self.toggle_button = ctk.CTkButton(
        #     self,
        #     image=self.off_img,
        #     text="",
        #     fg_color="transparent",
        #     height=50,
        #     width=50,
        #     anchor="center",
        #     hover_color=UIColors.White,
        #     command=self.update_toggle_state,
        # )
        #
        # self.toggle_button_label = ctk.CTkLabel(
        #     self,
        #     text="Display Mode",
        #     font=self.toggle_font,
        #     text_color=UIColors.Black,
        # )
        #
        # self.loading_label = ctk.CTkLabel(
        #     self,
        #     text="Loading",
        #     font=self.toggle_font,
        #     text_color=UIColors.Black,
        # )
        #
        # self.acquisition_label = ctk.CTkLabel(
        #     self,
        #     text="Acquisition",
        #     font=self.toggle_font,
        #     text_color=UIColors.Black,
        # )

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
        self.save_data_button.place(relx=self.controller.relx_pos(325), rely=self.controller.rely_pos(625), anchor=ctk.NW)
        self.reset_data_button.place(relx=self.controller.relx_pos(322), rely=self.controller.rely_pos(675), anchor=ctk.NW)
        selector_pos = (530, 50)
        self.save_selector.place(relx=self.controller.relx_pos(selector_pos[0]), rely=self.controller.rely_pos(selector_pos[1]), anchor=ctk.NW)
        self.save_selector_label.place(relx=self.controller.relx_pos(selector_pos[0] + 35), rely=self.controller.rely_pos(selector_pos[1] - 30), anchor=ctk.NW)
        self.load_button.place(relx=self.controller.relx_pos(selector_pos[0] + 35), rely=self.controller.rely_pos(selector_pos[1] + 120), anchor=ctk.NW)
        self.pause_loading_button.place(relx=self.controller.relx_pos(selector_pos[0] + 22), rely=self.controller.rely_pos(selector_pos[1] + 170), anchor=ctk.NW)
        self.position_label.place(relx=self.controller.relx_pos(540), rely=self.controller.rely_pos(350), anchor=ctk.NW)
        self.time_label.place(relx=self.controller.relx_pos(250), rely=self.controller.rely_pos(525), anchor=ctk.NW)
        # Rest of the Toggle setup
        # toggle_pos = (590, 30)
        # self.toggle_button.place(relx=self.controller.relx_pos(toggle_pos[0]), rely=self.controller.rely_pos(toggle_pos[1]), anchor=ctk.NW)
        # self.toggle_button_label.place(relx=self.controller.relx_pos(toggle_pos[0] - 20), rely=self.controller.rely_pos(toggle_pos[1] - 20), anchor=ctk.NW)
        # self.acquisition_label.place(relx=self.controller.relx_pos(toggle_pos[0] - 85),
        #                                rely=self.controller.rely_pos(toggle_pos[1] + 14), anchor=ctk.NW)
        # self.loading_label.place(relx=self.controller.relx_pos(toggle_pos[0] + 68),
        #                          rely=self.controller.rely_pos(toggle_pos[1] + 14), anchor=ctk.NW)
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
        x_dim = 240  # Define width
        y_dim = round(x_dim) + 2  # Define height
        pad_y = (350 - y_dim) // 2  # Center vertically by evenly distributing padding
        pad_x = (400 - x_dim) // 2  # Center horizontally by evenly distributing padding

        if self.plate_mask_cache is None:
            mask_img = Image.open(self.mask_path).rotate(30)  # Rotate as needed
            mask_img = mask_img.resize((x_dim, y_dim))  # Resize the mask
            mask_array = np.array(mask_img.convert("RGB"))
            mask_array = np.pad(
                mask_array,
                ((pad_y, pad_y), (pad_x, pad_x), (0, 0)),  # Even padding
                mode="constant",
            )
            self.plate_mask_cache = np.invert(mask_array.astype(bool))  # Adjust for logical inversion

        return self.plate_mask_cache

    def get_circular_mask_array(self):
        if self.circular_mask_cache is None:
            self.circular_mask_cache = np.ones((350, 400, 3), dtype=np.uint8)
            # rr, cc = disk((185, 200), 110)
            # self.circular_mask_cache[rr, cc, :] = 1
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

    def update_position_and_time_ui(self, x0, y0):
        if self.controller.reading_daq:
            position_text = "Current position:" + "\n" + f"x: {x0:.2f} mm" + "\n" + f"y: {y0:.2f} mm"
            if self.power_meter.start_time is None:
                acquisition_time = 0
            else:
                self.current_acquisition_time = time.time() - self.power_meter.start_time + self.elapsed_time
            time_text = "Durée de l'enregistrement: " + f"{self.current_acquisition_time:.2f} sec"
            self.position_label.configure(text=position_text)
            self.time_label.configure(text=time_text)

    def update_gradient(self):
        params = self.power_meter.get_laser_params()
        _, x0, y0, _, _ = params
        self.update_position_and_time_ui(x0, y0)
        Z = gaussian_2d((self.X, self.Y), *params)
        laser_position = (x0, y0)

        # Normalize the data to fit a colormap
        norm = mcolors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
        colormap = plt.get_cmap("coolwarm")

        # Convert the Gaussian values to RGB colors
        # img_array = (colormap(norm(Z))[:, :, :3] * 255).astype(np.uint8)
        img_array = np.ones((350, 400, 3), dtype=np.uint8) * UIColors.White.rgb_value
        img_array = self.apply_masks_to_gradient(img_array)

        # Convert to PIL Image
        img = Image.fromarray(img_array)

        # Convert to Tkinter-compatible format
        self.img_tk = ImageTk.PhotoImage(img)

        # Update the label's image
        self.canvas.itemconfig(self.image_id, image=self.img_tk)

        self.draw_overlay_shapes(norm, colormap, laser_position)

        # Schedule next update (simulate real-time data)
        self.canvas.after(20, self.update_gradient)  # Update every second

    def get_ui_thermistance_positions(self):
        positions = np.array(self.power_meter.xy_coords)
        r_out = self.power_meter.r_out
        positions[0, :] += r_out
        positions[1, :] = np.abs(positions[1, :] - r_out)
        positions /= r_out * 2
        positions[0, :] *= 200
        positions[1, :] *= 175
        positions[0, :] += 100
        positions[1, :] += 92.5
        return positions

    def convert_radial_to_ui_pos(self, pos):
        x, y = pos
        r_out = self.power_meter.r_out
        x += r_out
        y -= r_out
        x = x / (r_out * 2)
        y = abs(y) / (r_out * 2)
        x, y = (x * 200) + 100, (y * 175) + 92.5
        return x, y

    def draw_overlay_shapes(self, norm, colormap, current_pos):
        """Draw circles or other shapes on top of the heatmap"""
        self.canvas.delete("overlay")  # Remove previous shapes
        x, y = self.convert_radial_to_ui_pos(current_pos)
        temps = self.power_meter.get_temperature_values()
        positions = self.get_ui_thermistance_positions()
        self.canvas.create_oval(
            x - 10,
            y - 10,
            x + 10,
            y + 10,
            fill="Red",
            outline="Black",
            width=2,
            tags="overlay",
        )

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
    # Toggle command
    # def update_toggle_state(self):
    #     match self.toggle_state.get():
    #         case UIMode.Acquisition:
    #             self.toggle_state.set(UIMode.Loading)
    #             self.toggle_button.configure(image=self.on_img)
    #         case UIMode.Loading:
    #             self.toggle_state.set(UIMode.Acquisition)
    #             self.toggle_button.configure(image=self.off_img)

    def start_acquisition_daq(self):
        try:
            self.power_meter.start_acquisition()
        except RuntimeError:
            print("No Device Detected")
            return
        if self.activation_count == 0:
            self.read_daq_data()
            self.activation_count += 1
        elif self.activation_count >= 1:
            self.power_meter.start_time = time.time()
        self.start_acquisition_button.configure(state="disabled")
        self.stop_acquisition_button.configure(state="normal")
        self.daq_display_button.configure(state="normal")
        self.save_data_button.configure(state="disabled")
        self.reset_data_button.configure(state="disabled")
        self.controller.reading_daq = True

    def stop_acquisition_daq(self):
        self.controller.updating_plot = False
        self.controller.reading_daq = False
        self.elapsed_time = self.current_acquisition_time
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

    def start_loading_data(self):
        self.controller.reading_save = True
        self.power_meter.loader.load_index = 16
        self.read_loaded_data()
        self.load_button.configure(state="disabled")
        self.pause_loading_button.configure(state="normal")

    def pause_loading_data(self):
        self.controller.reading_save = False
        self.load_button.configure(state="normal")
        self.pause_loading_button.configure(state="disabled")

    def read_loaded_data(self):
        if self.controller.reading_save:
            save_to_load = self.save_selector.get()
            load_index = self.power_meter.loader.find_combobox_index(save_to_load)
            self.controller.lines = self.power_meter.fetch_simulation_data(load_index, self.controller.lines)
            if self.controller.updating_plot:
                self.controller.frames[DAQReadingsWindow].update_plot()
            self.after(10, self.read_loaded_data)


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
