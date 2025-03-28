import tkinter as tk
import customtkinter as ctk
import numpy as np
import time
import threading
from enum import StrEnum
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from packages.functions import gaussian_2d, PowerMeter
from PIL import Image, ImageTk
from pathlib import Path
from skimage.draw import disk

home_directory = Path(__file__).parents[0]


def setup_grid(self, rows: int, cols: int):
    """
    Sets up the grid layout to place the buttons automatically in the current frame.
    """
    for row in range(rows):
        self.grid_rowconfigure(row, weight=1, uniform="row")
    for col in range(cols):
        self.grid_columnconfigure(col, weight=1, uniform="col")


tk.Frame.setup_grid = setup_grid


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


class PowerMeterUI(tk.Tk):
    def __init__(self, *args, **kwargs):
        # Initializing the Tkinter window that will hold the app
        tk.Tk.__init__(self, *args, **kwargs)
        self.system_width, self.system_height = (
            self.winfo_screenwidth(),
            self.winfo_screenheight() - 70,
        )
        self.screen_dims = (self.system_width, self.system_height)
        self.geometry(f"{self.system_width}x{self.system_height}")
        self.minsize(self.system_width, self.system_height)
        self.maxsize(self.system_width, self.system_height)

        # creating a container
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # initializing frames to an empty dictionary
        self.frames = {}

        # iterating through the different frame layouts
        for F in (MainWindow, OtherWindow):
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


class MainWindow(tk.Frame):
    """
    Main window of the app, displays power (W), wavelength (nm) and heat map.
    """

    def __init__(self, master, controller):
        super().__init__(master)
        self.master = master
        self.controller = controller
        self.configure(bg=UIColors.White)
        self.setup_grid(6, 3)
        self.label_font = ctk.CTkFont(family="Times New Roman", size=20, weight="bold")
        self.text_font = ctk.CTkFont(family="Times New Roman", size=15)
        self.power_meter = PowerMeter()
        self.mask_path = home_directory / "Plate.png"
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
        )
        self.canvas = tk.Canvas(
            self, width=400, height=350, bg=UIColors.White, highlightthickness=0
        )
        self.canvas.place(x=175, y=250)
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW)

        self.power_txt_box.grid(row=0, column=1, padx=10, pady=10)
        self.power_txt_box_label.place(x=335, y=12)
        self.wavelength_txt_box.grid(row=1, column=1, padx=10, pady=10)
        self.wavelength_txt_box_label.place(x=300, y=135)
        self.acquisition_button.place(x=300, y=625)
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


class OtherWindow(tk.Frame):
    """
    Other Window in the app just in case. For now, not used for anything.
    """

    def __init__(self, master, controller):
        super().__init__(master)
        self.master = master
        self.controller = controller


if __name__ == "__main__":
    app = PowerMeterUI()
    app.title("Power Meter Interface v0.2.2")
    app.mainloop()
