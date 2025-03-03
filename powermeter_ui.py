import tkinter as tk
import customtkinter as ctk
import numpy as np
import time
import threading
from enum import StrEnum


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
        self.canvas = tk.Canvas(
            self, width=500, height=500, bg=UIColors.White, highlightthickness=0
        )
        self.canvas.place(x=175, y=300)
        # self.canvas.create_rectangle(0, 0, 550, 500, fill=UIColors.White, outline=UIColors.White)
        dims = (400, 350)
        self.canvas.create_polygon(
            (dims[0] / 4, 0),
            ((dims[0] / 4) * 3, 0),
            (dims[0], dims[1] / 2),
            ((dims[0] / 4) * 3, dims[1]),
            (dims[0] / 4, dims[1]),
            (0, dims[1] / 2),
            fill=UIColors.White,
            outline="black",
            width=2,
        )

        x, y = dims[0] / 2, dims[1] / 2
        positions = np.load("positions_list.npy")
        for position in positions:
            x, y = position
            self.canvas.create_oval(x, y, x + 5, y + 5, fill="red", outline="")

        self.power_txt_box.grid(row=0, column=1, padx=10, pady=10)
        self.power_txt_box_label.place(x=335, y=12)
        self.wavelength_txt_box.grid(row=1, column=1, padx=10, pady=10)
        self.wavelength_txt_box_label.place(x=300, y=135)
        threading.Thread(target=self.update_values).start()

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
