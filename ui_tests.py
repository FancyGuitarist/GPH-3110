import tkinter as tk
from tkinter import ttk
import threading
import time
import numpy as np
import nidaqmx
import numpy as np
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.constants import AcquisitionType, LineGrouping
import time
from nidaqmx.constants import TerminalConfiguration
from scipy.optimize import curve_fit
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import matplotlib.pyplot as plt


def gaussian_2d(coords, A, x0, y0, sigma_X, sigma_Y):
    x, y = coords
    return A * np.exp(-((x - x0) ** 2 / (2 * sigma_X ** 2) + (y - y0) ** 2 / (2 * sigma_Y ** 2)))


def polar_to_cartesian(r, theta_deg):
    theta = np.radians(theta_deg)
    return [r * np.cos(theta), r * np.sin(theta)]


class PowerMeterApp:
    def __init__(self, root):

        # daq
        self.DAQ_DEVICE = "Daddy"  # Replace with your actual DAQ device name
        self.SAMPLES_PER_READ = 10
        self.SAMPLE_RATE = 10000  # Hz

        # scaling de la position
        self.scale_factor = 1.8

        # prediction puissance
        self.predict = True
        self.N = 10  # mettre plus haut pour moins de bruit
        self.estimate_factor = 6  # mettre plus haut pour une prediction plus rapide (mais si c'Est trop haut, ca overshoot)

        # facteur longueur d'onde
        # si on met un echelon de 5W, et on voit 4W, on prend le facteur et on le multiplie par (4W/5W = 0.😎
        self.facteur_976 = 3.3
        self.facteur_1976 = 2
        self.facteur_450 = 3.6

        self.PRAY = True

        self.data = np.zeros((5, self.SAMPLES_PER_READ))
        self.p0 = [0, 0.0, 0.0, 8.5, 8.5]
        self.power_real_cache = []
        self.time_real_cache = []

        r1 = 5.0  # example radius
        r2 = 13.97

        self.positions = []
        for angle in [30, 90, 150, 210, 270, 330]:
            self.positions.append(polar_to_cartesian(r2, angle))

        self.positions = np.array(self.positions)

        rotation_angle = np.radians(0)
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)]
        ])
        self.positions = np.dot(self.positions, rotation_matrix.T)

        self.root = root
        self.root.title("Power Meter")

        # Set fixed window size and disable resizing
        self.root.geometry("1200x800")
        self.root.resizable(False, False)

        # Add Quebec Watt App logo at the top right corner

        # Create a label to display the calculated power
        self.power_cache = []
        self.power_label = ttk.Label(self.root, text="Puissance: 0 W", font=("Arial", 24))
        self.power_label.place(relx=0.01, rely=0.4, anchor="w")
        self.update_power(0)  # Initialize with 0 W

        self.wl_cache = []
        self.current_wl = 976
        self.wl_label = ttk.Label(self.root, text="Longueur d'onde: 975 nm", font=("Arial", 24))
        self.wl_label.place(relx=0.01, rely=0.45, anchor="w")
        self.update_wavelength(0)

        self.x_cache = []
        self.y_cache = []
        self.position_label = ttk.Label(self.root, text="Position : x = 0, y = 0", font=("Arial", 24))
        self.position_label.place(relx=0.01, rely=0.5, anchor="w")
        self.update_position(0, 0)

        self.time_cache = []

        self.size = 100
        # Create a canvas to draw the circle
        self.canvas = tk.Canvas(self.root, width=2 * self.size + 10, height=2 * self.size + 10, highlightthickness=0,
                                bg=self.root.cget("bg"))
        self.canvas.place(relx=0.2, rely=0.7, anchor="center")

        # Draw a circle in the center of the canvas
        canvas_width = 2 * self.size + 10
        canvas_height = 2 * self.size + 10
        circle_radius = self.size
        self.canvas.create_oval(
            (canvas_width // 2) - circle_radius,
            (canvas_height // 2) - circle_radius,
            (canvas_width // 2) + circle_radius,
            (canvas_height // 2) + circle_radius,
            outline="blue"
        )

        # Create a red dot inside the circle
        self.dot_radius = 7
        self.dot = self.canvas.create_oval(
            (canvas_width // 2) - self.dot_radius,
            (canvas_height // 2) - self.dot_radius,
            (canvas_width // 2) + self.dot_radius,
            (canvas_height // 2) + self.dot_radius,
            fill="red",
            outline="red"
        )

        # Create a button to start acquisition
        self.start_button = ttk.Button(self.root, text="Commencer l'acquisition", command=self.start_acquisition)
        self.start_button.place(relx=0.2, rely=0.87, anchor="center")
        self.start_button.config(style="Green.TButton")

        # Define a custom style for the green button with larger font and padding
        style = ttk.Style()
        style.configure("Green.TButton", background="green", font=("Arial", 18), padding=10)
        style.map("Green.TButton", foreground=[("!disabled", "green")])

        # Create a text field for entering the wavelength
        self.wavelength_entry_label = ttk.Label(self.root, text="Longueur d'onde (nm):", font=("Arial", 24))
        self.wavelength_entry_label.place(relx=0.87, rely=0.1, anchor="e")

        self.wavelength_entry = ttk.Entry(self.root, font=("Arial", 24), width=6)
        self.wavelength_entry.place(relx=0.87, rely=0.1, anchor="w")

        # Add a button to update the wavelength
        self.update_wavelength_button = ttk.Button(self.root, text="Confirmer", command=self.current_wavelength)
        self.update_wavelength_button.place(relx=0.85, rely=0.18, anchor="w")
        self.update_wavelength_button.config(style="Large.TButton")

        # Define a custom style for the button with larger font and padding
        style = ttk.Style()
        style.configure("Large.TButton", font=("Arial", 16), padding=10)

        # Create a frame for the real-time plot
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.place(relx=1, rely=1, anchor="se")

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        # self.fig.tight_layout()  # Adjust layout to prevent clipping
        self.ax.set_title("Puissance en temps réel")
        self.ax.set_xlabel("Temps (s)")
        self.ax.set_ylabel("Puissance (W)")
        self.line, = self.ax.plot([], [], 'r-')  # Initialize an empty line

        # Embed the matplotlib figure into the Tkinter frame
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_plot.get_tk_widget().pack()

        # Initialize data for the plot
        self.plot_x_data = []
        self.plot_y_data = []

        # Create a text box for displaying error messages
        self.error_text_box = tk.Text(self.root, height=5, width=30, font=("Arial", 16), wrap="word", state="disabled",
                                      bg=self.root.cget("bg"))
        self.error_text_box.place(relx=0.98, rely=0.4, anchor="e")

        # Add a text next to the mode label
        self.mode_status_label = ttk.Label(self.root, text="Mode longueur d\'onde", font=("Arial", 24))
        self.mode_status_label.place(relx=0.58, rely=0.01, anchor="nw")
        # Create a label for the mode toggle button
        self.mode_label = ttk.Label(self.root, text="Mode:", font=("Arial", 24))
        self.mode_label.place(relx=1, rely=0, anchor="ne")

        # Create a toggle button for automatic/manual mode
        self.mode_var = tk.BooleanVar(value=True)  # True for Automatic, False for Manual
        self.mode_button = ttk.Button(self.root, text="Automatique", command=self.toggle_mode)
        self.mode_button.place(relx=1, rely=0, anchor="ne")
        self.mode_button.config(style="Toggle.TButton")

        self.wavelength_entry.place_forget()
        self.wavelength_entry_label.place_forget()
        self.update_wavelength_button.place_forget()

        # Define a custom style for the toggle button
        style = ttk.Style()
        style.configure("Toggle.TButton", font=("Arial", 16), padding=10)

        try:
            # Load the logo image
            logo_image = Image.open("quebecwattapplogo.icns").resize((100, 100), Image.Resampling.LANCZOS)
            logo_photo = ImageTk.PhotoImage(logo_image)

            # Create a label to display the logo
            # Resize the logo to make it bigger
            logo_image = logo_image.resize((250, 250), Image.Resampling.LANCZOS)
            logo_photo = ImageTk.PhotoImage(logo_image)

            self.logo_label = tk.Label(self.root, image=logo_photo, bg=self.root.cget("bg"))
            self.logo_label.image = logo_photo  # Keep a reference to avoid garbage collection
            self.logo_label.place(relx=0.01, rely=0.01, anchor="nw")
        except Exception as e:
            self.display_error_message(f"Erreur de chargement du logo : {e}")

    def toggle_mode(self):
        """Toggle between automatic and manual mode."""
        self.mode_var.set(not self.mode_var.get())
        if self.mode_var.get():
            self.mode_button.config(text="Automatique")
            self.wavelength_entry.place_forget()
            self.wavelength_entry_label.place_forget()
            self.update_wavelength_button.place_forget()
            self.display_error_message("Mode automatique activé.")
        else:
            self.mode_button.config(text="Manuel")
            self.wavelength_entry.place(relx=0.87, rely=0.1, anchor="w")
            self.wavelength_entry_label.place(relx=0.87, rely=0.1, anchor="e")
            self.update_wavelength_button.place(relx=0.85, rely=0.18, anchor="w")

            self.display_error_message("Mode manuel activé.")

    def display_error_message(self, message):
        """Display an error message in the text box."""
        self.error_text_box.config(state="normal")
        self.error_text_box.delete(1.0, tk.END)  # Clear previous messages
        self.error_text_box.insert(tk.END, message)
        self.error_text_box.config(state="disabled")

    def clear_plot(self):
        """Clear the plot data."""
        self.plot_x_data.clear()
        self.plot_y_data.clear()
        self.line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()

    def update_plot(self, x, y):
        """Update the real-time plot with new x and y values."""
        self.plot_x_data.append(x)
        self.plot_y_data.append(y)

        # Limit the number of points displayed
        if len(self.plot_x_data) > 100:
            self.plot_x_data.pop(0)
            self.plot_y_data.pop(0)

        # Update the line data
        self.line.set_data(self.plot_x_data, self.plot_y_data)
        self.ax.relim()  # Recalculate limits
        self.ax.autoscale_view()  # Autoscale the view
        self.canvas_plot.draw()  # Redraw the canvas

    def current_wavelength(self):
        try:
            self.current_wl = max(0, min(2500, int(self.wavelength_entry.get())))

            self.wl_label.config(text=f"Longueur d'onde: {self.current_wl} nm")
        except:
            # print("Womp Womp")
            pass

    def update_power(self, power):
        """Update the power label with the given power value."""
        power = round(power, 2)
        self.power_cache.append(power)
        self.power_label.config(text=f"Puissance: {power} W")

    def update_wavelength(self, wl):
        """Update the wavelength label with the given wavelength value."""
        wl = round(wl, 2)
        self.wl_cache.append(wl)
        self.wl_label.config(text=f"Longueur d'onde: {wl} nm")

    def update_position(self, x, y):
        """Update the position label with the given x and y values."""
        x = round(x, 2)
        y = round(y, 2)
        self.x_cache.append(x)
        self.y_cache.append(y)
        self.position_label.config(text=f"Position : x = {x}, y = {y}")

    def update_time(self, time):
        """Update the time label with the given time value."""
        time = round(time, 2)
        self.time_cache.append(time)

    def update_dot_position(self, x, y):
        """Update the position of the red dot inside the circle."""
        y = -1 * y
        canvas_width = 2 * self.size + 10
        canvas_height = 2 * self.size + 10
        circle_radius = self.size

        # Ensure the dot stays within the circle
        if (x ** 2 + y ** 2) > circle_radius ** 2:
            return

        # Update the dot's position
        self.canvas.coords(
            self.dot,
            (canvas_width // 2) + x - self.dot_radius,
            (canvas_height // 2) + y - self.dot_radius,
            (canvas_width // 2) + x + self.dot_radius,
            (canvas_height // 2) + y + self.dot_radius
        )

    def end_acquisition(self):

        """Handle the end acquisition button click."""
        self.acquisition_running = False
        self.start_button.config(state="normal")

        if hasattr(self, 'end_button'):
            self.end_button.destroy()

        # Show the start button again
        self.start_button.place(relx=0.2, rely=0.87, anchor="center")

        # Create a button to save cached data
        self.save_button = ttk.Button(self.root, text="Sauvegarder les données", command=self.save_cached_data)
        self.save_button.place(relx=0.2, rely=0.98, anchor="s")
        self.save_button.config(style="Blue.TButton")

        # Define a custom style for the blue button with larger font and padding
        style = ttk.Style()
        style.configure("Blue.TButton", background="blue", font=("Arial", 18), padding=10)
        style.map("Blue.TButton", foreground=[("!disabled", "blue")])

        try:
            self.task.close()
            self.do_task.close()
        except:
            pass

    def save_cached_data(self):
        """Save the cached data to a file with a user-specified name."""

        def save_file():
            filename = file_name_entry.get()
            if filename:
                with open(f"{filename}.txt", "w") as f:
                    # Write the header
                    f.write("Temps(s),Puissance(W),X(mm),Y(mm),Longueur d\'onde(nm)\n")
                    for time, power, wl, x, y in zip(self.time_cache, self.power_cache, self.wl_cache, self.x_cache,
                                                     self.y_cache):
                        f.write(f"{time},{wl},{x},{y},{power}\n")
                print(f"Data saved to {filename}.txt")
                popup.destroy()

        # Create a popup window
        popup = tk.Toplevel(self.root)
        popup.title("Nom du fichier à sauvegarder")
        popup.geometry("600x300")
        popup.resizable(False, False)

        # Add a label and entry field for the file name
        tk.Label(popup, text="Entrée le nom du fichier:", font=("Arial", 30)).pack(pady=10)
        file_name_entry = ttk.Entry(popup, font=("Arial", 30))
        file_name_entry.pack(pady=5)

        # Add a save button
        save_button = ttk.Button(popup, text="Enregistrer", command=save_file)
        save_button.pack(pady=10)
        save_button.config(style="Large.TButton")

        self.display_error_message("Fichié sauvegardé avec succès !")

    def start_acquisition(self):
        """Start the acquisition process."""
        # Start the acquisition thread
        self.start_button.place_forget()
        self.clear_plot()
        self.acquisition_running = True
        self.acquisition_thread = threading.Thread(target=self.acquisition_task, daemon=True)
        self.acquisition_thread.start()

        # Disable the start button and create an end button
        self.start_button.config(state="disabled")
        self.end_button = ttk.Button(self.root, text="Arrêter l'acquisition", command=self.end_acquisition)
        self.end_button.place(relx=0.2, rely=0.87, anchor="center")
        self.end_button.config(style="Red.TButton")

        # Define a custom style for the green button with larger font and padding
        style = ttk.Style()
        style.configure("Red.TButton", background="red", font=("Arial", 18), padding=10)
        style.map("Red.TButton", foreground=[("!disabled", "red")])

        if hasattr(self, 'save_button'):
            self.save_button.destroy()

    def acquisition_task(self):
        """Simulate a long-running acquisition task."""
        self.display_error_message("")
        i = 0
        cache = np.zeros((5, 16))
        start_time = time.time()
        self.time_cache = []
        self.power_cache = []
        self.wl_cache = []
        self.x_cache = []
        self.y_cache = []
        self.power_real_cache = []
        self.time_real_cache = []
        try:
            self.task = nidaqmx.Task()
            self.task.ai_channels.add_ai_voltage_chan(f"{self.DAQ_DEVICE}/ai0:4",
                                                      terminal_config=TerminalConfiguration.RSE)
            self.task.timing.cfg_samp_clk_timing(rate=self.SAMPLE_RATE, sample_mode=AcquisitionType.FINITE,
                                                 samps_per_chan=self.SAMPLES_PER_READ)
            self.reader = AnalogMultiChannelReader(self.task.in_stream)
            self.do_task = nidaqmx.Task()
            self.do_task.do_channels.add_do_chan(
                f"{self.DAQ_DEVICE}/port0/line0:3",  # Replace with your device and lines
                line_grouping=LineGrouping.CHAN_PER_LINE
            )
            while self.acquisition_running:
                if self.PRAY:
                    if self.task.is_task_done():
                        bits = [bool(int(b)) for b in format(i, '04b')]
                        self.do_task.write(bits)
                        if self.do_task.is_task_done():
                            self.reader.read_many_sample(self.data, number_of_samples_per_channel=self.SAMPLES_PER_READ)
                            cache[:, i] = np.mean(self.data, axis=1)

                    if i == 15:
                        print("Cache : ", cache)
                        THref = cache[4, 1]
                        # TH1 = np.mean(cache[0,0:15])
                        # TH2 = np.mean(cache[1,0:15])
                        # TH3 = np.mean(cache[2,0:15])
                        TH4 = cache[4, 3]
                        TH5 = cache[4, 5]
                        TH6 = cache[4, 7]
                        TH7 = cache[4, 9]
                        TH8 = cache[4, 11]
                        TH9 = cache[4, 13]
                        T_measured = np.array(
                            [TH9 - THref, TH4 - THref, TH8 - THref, TH6 - THref, TH7 - THref, TH5 - THref])
                        # print(T_measured)
                        x_data, y_data = self.positions[:, 0], self.positions[:, 1]
                        try:
                            popt, _ = curve_fit(gaussian_2d, (x_data, y_data), T_measured, p0=self.p0,
                                                bounds=([0, -15, -15, 8, 8], [60, 15, 15, 9, 9]), maxfev=2000)
                        except:
                            popt = self.p0

                        if popt[0] < 1:
                            popt = self.p0

                        A, x0, y0, _, _ = popt
                        x0 = x0 * self.scale_factor
                        y0 = y0 * self.scale_factor

                        # print(popt)

                        if self.current_wl > 960 and self.current_wl < 990:
                            self.power_real_cache.append(A / self.facteur_976)
                        elif self.current_wl > 1960 and self.current_wl < 1990:
                            self.power_real_cache.append(A / self.facteur_1976)
                        elif self.current_wl > 430 and self.current_wl < 460:
                            self.power_real_cache.append(A / self.facteur_450)

                        current_time = round(time.time() - start_time, 2)
                        self.time_real_cache.append(current_time)

                        if len(self.power_real_cache) > self.N:
                            self.power_real_cache.pop(0)
                            self.time_real_cache.pop(0)

                        if self.predict:
                            if len(self.power_real_cache) > 1:
                                dP_dt = np.mean(np.diff(self.power_real_cache) / np.diff(self.time_real_cache))
                                puissance_predite = dP_dt * self.estimate_factor + np.mean(self.power_real_cache)
                            else:
                                self.update_power(0)
                                puissance_predite = 0

                            self.update_plot(current_time, puissance_predite)
                            self.update_power(puissance_predite)
                        else:
                            self.update_plot(current_time, self.power_real_cache[-1])
                            self.update_power(self.power_real_cache[-1])

                        self.update_time(current_time)
                        self.update_dot_position(x0 * 5, y0 * 5)
                        self.update_position(x0, y0)
                        self.update_wavelength(self.current_wl)

                    i = (i + 1) % 16
        except:
            self.display_error_message(
                "Erreur de communication avec le DAQ. Vérifiez la connexion et le nom de l'appareil.")
            self.end_acquisition()


if _name_ == "_main_":
    root = tk.Tk()
    app = PowerMeterApp(root)

    root.mainloop()