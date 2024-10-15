import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from keras.models import load_model
from data.dataSCATSMap import process_data, prepare_model_data, create_traffic_map
from src.route_guidance import route_guidance
from src.utils import (
    format_prediction_result,
    format_route_result,
)
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import dates as mdates
from datetime import datetime
import webbrowser
import tempfile
import logging
from PIL import Image, ImageTk
import io
import yaml
import folium
from folium.plugins import MarkerCluster

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load configuration
with open("config.yml", "r") as config_file:
    config = yaml.safe_load(config_file)


class TFPSGUI:
    def __init__(self, master):
        self.master = master
        master.title("Traffic Flow Prediction System")
        master.geometry(config["gui"]["window_size"])

        self.models = {}
        self.available_scats = {}
        self.df = None
        self.scaler = None
        self.street_segments = None
        self.model_data = None
        self.map = None
        self.map_file = None
        self.data_loaded = False

        self.create_widgets()
        self.setup_data()

    def setup_data(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(
                current_dir, "..", "data", config["data"]["file_path"]
            )
            logger.info(f"Attempting to process data from: {file_path}")

            self.n_scats = tk.StringVar(value="all")
            result = process_data(file_path, n_scats=None)

            if isinstance(result, tuple) and len(result) == 3:
                self.df, self.scaler, self.street_segments = result
                self.model_data = prepare_model_data(self.df)
                self.setup_models()
                self.update_dropdowns()
                self.data_loaded = True
            else:
                raise ValueError(f"Unexpected result from process_data: {result}")

        except Exception as e:
            logger.error(f"Error setting up data: {e}", exc_info=True)
            self.show_error_message(
                f"Failed to load data: {str(e)}\nPlease check the console for more information."
            )

    def setup_models(self):
        for name in config["training"]["models"]:
            try:
                self.models[name] = {}
                self.available_scats[name] = set()
                for scats in self.df["SCATS Number"].unique():
                    model_path = os.path.join("model", "trained", f"{name}_{scats}.h5")
                    if os.path.exists(model_path):
                        self.models[name][scats] = load_model(model_path)
                        self.available_scats[name].add(scats)
                    else:
                        logger.warning(f"Model file not found: {model_path}")
            except Exception as e:
                logger.error(f"Error loading {name} models: {e}")

        self.update_model_dropdown()
        self.update_street_options()

    def update_dropdowns(self):
        if self.df is not None:
            scats_numbers = sorted(self.df["SCATS Number"].unique())
            self.origin_dropdown["values"] = scats_numbers
            self.destination_dropdown["values"] = scats_numbers

            dates = sorted(self.df["Date"].dt.date.unique().astype(str))
            self.date_dropdown["values"] = dates
            if dates:
                self.date_var.set(dates[0])

    def update_model_dropdown(self):
        available_models = [name for name in self.models if self.models[name]]
        self.model_dropdown["values"] = available_models
        if available_models:
            self.model_var.set(available_models[0])
        else:
            self.model_var.set("")
            messagebox.showwarning(
                "No Models",
                "No trained models found. Please train models before using the application.",
            )

    def update_street_options(self, *args):
        model_name = self.model_var.get().lower()
        if not model_name:
            self.street_dropdown["values"] = []
            self.street_var.set("")
            return

        available_scats = self.available_scats.get(model_name, set())
        available_streets = sorted(
            self.df[self.df["SCATS Number"].isin(available_scats)]["Street"].unique()
        )

        self.street_dropdown["values"] = available_streets
        if available_streets:
            self.street_var.set(available_streets[0])
        else:
            self.street_var.set("")
            messagebox.showwarning(
                "No Streets",
                f"No streets available for the selected model: {model_name}",
            )

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.create_prediction_tab()
        self.create_route_guidance_map_tab()

    def create_prediction_tab(self):
        pred_frame = ttk.Frame(self.notebook)
        self.notebook.add(pred_frame, text="Traffic Prediction")

        input_frame = ttk.LabelFrame(pred_frame, text="Prediction Inputs")
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(input_frame, text="Select Model:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(
            input_frame, textvariable=self.model_var, state="readonly"
        )
        self.model_dropdown.grid(row=0, column=1, padx=5, pady=5)
        self.model_dropdown.bind("<<ComboboxSelected>>", self.update_street_options)

        ttk.Label(input_frame, text="Select Street:").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        self.street_var = tk.StringVar()
        self.street_dropdown = ttk.Combobox(
            input_frame, textvariable=self.street_var, state="readonly"
        )
        self.street_dropdown.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Select Date:").grid(
            row=2, column=0, padx=5, pady=5, sticky="w"
        )
        self.date_var = tk.StringVar()
        self.date_dropdown = ttk.Combobox(
            input_frame, textvariable=self.date_var, state="readonly"
        )
        self.date_dropdown.grid(row=2, column=1, padx=5, pady=5)

        ttk.Button(input_frame, text="Predict", command=self.predict).grid(
            row=3, column=0, columnspan=2, pady=10
        )

        result_frame = ttk.LabelFrame(pred_frame, text="Prediction Results")
        result_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.result_text = tk.Text(
            result_frame, height=5, width=40, wrap=tk.WORD, font=("Arial", 10)
        )
        self.result_text.pack(pady=5, padx=5, fill=tk.X)

        self.graph_frame = ttk.Frame(result_frame)
        self.graph_frame.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

    def create_route_guidance_map_tab(self):
        combined_frame = ttk.Frame(self.notebook)
        self.notebook.add(combined_frame, text="Map & Route Guidance")

        # Left panel for inputs and results
        left_panel = ttk.Frame(combined_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Route guidance inputs
        input_frame = ttk.LabelFrame(left_panel, text="Route Inputs")
        input_frame.pack(fill=tk.X, pady=10)

        ttk.Label(input_frame, text="Origin SCATS:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.origin_var = tk.StringVar()
        self.origin_dropdown = ttk.Combobox(input_frame, textvariable=self.origin_var, state="readonly")
        self.origin_dropdown.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Destination SCATS:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.destination_var = tk.StringVar()
        self.destination_dropdown = ttk.Combobox(input_frame, textvariable=self.destination_var, state="readonly")
        self.destination_dropdown.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(input_frame, text="Find Routes", command=self.find_and_display_routes).grid(row=2, column=0, columnspan=2, pady=10)

        # Route results
        result_frame = ttk.LabelFrame(left_panel, text="Route Results")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.route_text = tk.Text(result_frame, height=20, width=40, wrap=tk.WORD, font=("Arial", 10))
        self.route_text.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

        # Right panel for map
        self.map_frame = ttk.Frame(combined_frame)
        self.map_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.show_map()

    def predict(self):
        try:
            model_name = self.model_var.get().lower()
            street = self.street_var.get()
            date = self.date_var.get()

            if not model_name:
                raise ValueError("Please select a model")

            street_data = self.df[self.df["Street"] == street]
            available_scats = street_data[
                street_data["SCATS Number"].isin(self.available_scats[model_name])
            ]

            if available_scats.empty:
                raise ValueError(
                    f"No trained model found for {model_name} on the selected street"
                )

            scats_number = available_scats["SCATS Number"].iloc[0]
            model = self.models[model_name][scats_number]

            selected_date = pd.to_datetime(date).date()
            day_data = available_scats[available_scats["Date"].dt.date == selected_date]

            if day_data.empty:
                raise ValueError("No data available for the selected date")

            # Ensure we have 96 time points for the full day
            full_day_times = pd.date_range(start=f"{date} 00:00", end=f"{date} 23:45", freq="15min")
            predictions = []
            actuals = day_data["TrafficVolume"].tolist()

            # Pad actuals with NaN if less than 96 points
            actuals += [np.nan] * (96 - len(actuals))

            for time in full_day_times:
                matching_data = day_data[day_data["DateTime"] == time]
                
                if matching_data.empty:
                    # If no data for this time, use the last available data point
                    last_known_data = day_data[day_data["DateTime"] < time].iloc[-1] if not day_data[day_data["DateTime"] < time].empty else None
                    
                    if last_known_data is not None:
                        input_data = self.prepare_input_data(model_name, last_known_data, available_scats)
                    else:
                        # If no previous data, use a default input (e.g., zeros)
                        input_shape = model.input_shape[1:]
                        input_data = np.zeros((1,) + input_shape)
                else:
                    input_data = self.prepare_input_data(model_name, matching_data.iloc[0], available_scats)

                prediction = model.predict(input_data)
                prediction = self.scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
                predictions.append(prediction)

            self.plot_prediction(street, date, day_data, predictions, actuals)

        except Exception as e:
            logger.exception(f"Prediction error: {str(e)}")
            messagebox.showerror("Prediction Error", str(e))

    def prepare_input_data(self, model_name, data_point, available_scats):
        scats_number = data_point["SCATS Number"]
        X_test = self.model_data[scats_number]["X_test"]
        
        sequence_length = X_test.shape[1]
        
        time_index = available_scats.index.get_loc(data_point.name)
        
        input_sequence = available_scats.iloc[max(0, time_index - sequence_length + 1):time_index + 1]['NormalizedVolume'].values
        
        if len(input_sequence) < sequence_length:
            input_sequence = np.pad(input_sequence, (sequence_length - len(input_sequence), 0), 'constant')
        
        if model_name in ["lstm", "gru", "bilstm", "cnnlstm"]:
            return np.reshape(input_sequence, (1, sequence_length, 1))
        else:  # SAES
            return np.reshape(input_sequence, (1, sequence_length))
            
    def plot_prediction(self, street, date, day_data, predictions, actuals):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(111)
        
        # Create a single time range for both predictions and actuals
        times = pd.date_range(start=f"{date} 00:00", end=f"{date} 23:45", freq="15min")
        
        # Ensure predictions and actuals have the same length as times
        predictions = predictions[:len(times)]
        actuals = actuals[:len(times)]

        # Remove NaN values from actuals and corresponding times and predictions
        valid_indices = ~np.isnan(actuals)
        plot_times = times[valid_indices]
        plot_actuals = np.array(actuals)[valid_indices]
        plot_predictions = np.array(predictions)[valid_indices]

        actual_line, = ax.plot(plot_times, plot_actuals, label="Actual", color="blue", alpha=0.7)
        predicted_line, = ax.plot(plot_times, plot_predictions, label="Predicted", color="red", alpha=0.7)

        ax.set_xlabel("Time of Day")
        ax.set_ylabel("Traffic Volume (vehicles per 15 min)")
        ax.set_title(f'24-Hour Traffic Volume Prediction - {street} ({date})')
        ax.legend()

        # Format x-axis to show hours
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        fig.autofmt_xdate()  # Rotate and align the tick labels

        # Set x-axis limits to ensure full day is shown
        ax.set_xlim(times[0], times[-1])

        # Set y-axis limits with some padding
        y_min = min(min(plot_actuals), min(plot_predictions))
        y_max = max(max(plot_actuals), max(plot_predictions))
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        # Highlight rush hours
        self.highlight_time_range(ax, "07:00", "10:00", "Morning Rush")
        self.highlight_time_range(ax, "16:00", "19:00", "Evening Rush")

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add navigation toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.graph_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.add_tooltip(fig, ax, actual_line, predicted_line)

    def add_tooltip(self, fig, ax, actual_line, predicted_line):
        tooltip = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        tooltip.set_visible(False)

        def update_tooltip(event):
            if event.inaxes == ax:
                cont_a, ind_a = actual_line.contains(event)
                cont_p, ind_p = predicted_line.contains(event)
                if cont_a:
                    pos = actual_line.get_xydata()[ind_a["ind"][0]]
                    tooltip.xy = pos
                    tooltip.set_text(f"Time: {mdates.num2date(pos[0]).strftime('%H:%M')}\nActual: {pos[1]:.2f}")
                    tooltip.set_visible(True)
                elif cont_p:
                    pos = predicted_line.get_xydata()[ind_p["ind"][0]]
                    tooltip.xy = pos
                    tooltip.set_text(f"Time: {mdates.num2date(pos[0]).strftime('%H:%M')}\nPredicted: {pos[1]:.2f}")
                    tooltip.set_visible(True)
                else:
                    tooltip.set_visible(False)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", update_tooltip)
        
    def highlight_time_range(self, ax, start, end, label):
        date = mdates.num2date(ax.get_xlim()[0]).date()
        start_time = pd.to_datetime(f"{date} {start}")
        end_time = pd.to_datetime(f"{date} {end}")
        
        ax.axvspan(start_time, end_time, alpha=0.2, color="yellow")
        ax.text(
            start_time,
            ax.get_ylim()[1],
            label,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )
        
    def find_routes(self):
        try:
            origin = self.origin_var.get()
            destination = self.destination_var.get()

            if not origin or not destination:
                messagebox.showerror(
                    "Input Error",
                    "Please select both origin and destination SCATS numbers.",
                )
                return

            routes = route_guidance(self.df, self.street_segments, origin, destination)

            self.route_text.config(state=tk.NORMAL)
            self.route_text.delete("1.0", tk.END)
            self.route_text.insert(tk.END, format_route_result(routes))
            self.route_text.config(state=tk.DISABLED)

        except ValueError as e:
            logger.exception(f"Route finding error: {str(e)}")
            messagebox.showerror("Route Finding Error", str(e))
        except Exception as e:
            logger.exception(f"Unexpected error in route finding: {str(e)}")
            messagebox.showerror(
                "Unexpected Error", f"An unexpected error occurred: {str(e)}"
            )

    def show_map(self):
        if not self.data_loaded:
            messagebox.showerror(
                "Data Not Loaded",
                "Data has not been loaded. Please check the console for errors.",
            )
            return 
        try:
            if self.map is None:
                if self.df is None or self.street_segments is None:
                    raise ValueError("Data not loaded. Please check the console for errors.")
                self.map = create_traffic_map(self.df, self.street_segments)

            if self.map_file:
                os.unlink(self.map_file)

            _, self.map_file = tempfile.mkstemp(suffix='.html')
            self.map.save(self.map_file)
            webbrowser.open('file://' + self.map_file)

        except Exception as e:
            logger.exception(f"Error creating map: {str(e)}")
            messagebox.showerror("Map Creation Error", f"An error occurred while creating the map: {str(e)}")

    def find_and_display_routes(self):
        if not self.data_loaded:
            messagebox.showerror("Data Not Loaded", "Data has not been loaded. Please check the console for errors.")
            return
        try:
            origin = self.origin_var.get()
            destination = self.destination_var.get()

            if not origin or not destination:
                messagebox.showerror("Input Error", "Please select both origin and destination SCATS numbers.")
                return

            routes = route_guidance(self.df, self.street_segments, origin, destination)

            self.route_text.config(state=tk.NORMAL)
            self.route_text.delete("1.0", tk.END)
            self.route_text.insert(tk.END, format_route_result(routes))
            self.route_text.config(state=tk.DISABLED)

            self.update_map_with_route(routes[0]['path'])  # Display the first (best) route on the map

        except ValueError as e:
            logger.exception(f"Route finding error: {str(e)}")
            messagebox.showerror("Route Finding Error", str(e))
        except Exception as e:
            logger.exception(f"Unexpected error in route finding: {str(e)}")
            messagebox.showerror("Unexpected Error", f"An unexpected error occurred: {str(e)}")

    def update_map_with_route(self, route):
        if self.map is None:
            self.show_map()

        # Add route to the map
        route_coords = [self.df[self.df['SCATS Number'] == int(scats)][['NB_LATITUDE', 'NB_LONGITUDE']].iloc[0].tolist() for scats in route]
        folium.PolyLine(locations=route_coords, color="blue", weight=4, opacity=0.8).add_to(self.map)

        # Save and display the updated map
        if self.map_file:
            os.unlink(self.map_file)

        _, self.map_file = tempfile.mkstemp(suffix='.html')
        self.map.save(self.map_file)
        webbrowser.open('file://' + self.map_file)

    def show_error_message(self, message):
        error_frame = ttk.Frame(self.master)
        error_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        error_label = ttk.Label(
            error_frame,
            text=message,
            foreground="red",
            wraplength=600,
            justify=tk.CENTER,
        )
        error_label.pack(expand=True)

        ttk.Button(error_frame, text="Exit", command=self.master.quit).pack(pady=10)


def main():
    root = tk.Tk()
    gui = TFPSGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

def __del__(self):
        if self.map_file and os.path.exists(self.map_file):
            os.unlink(self.map_file)
