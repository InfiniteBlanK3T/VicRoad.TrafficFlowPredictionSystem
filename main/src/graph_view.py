import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from keras.models import load_model
from data.dataSCATSMap import process_data, prepare_model_data, create_traffic_map
from src.route_guidance import route_guidance
from src.utils import format_prediction_result, plot_prediction, highlight_time_range, format_route_result
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import webbrowser
import logging
from PIL import Image, ImageTk
import io
import yaml

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load configuration
with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

class TFPSGUI:
    def __init__(self, master):
        self.master = master
        master.title("Traffic Flow Prediction System")
        master.geometry(config['gui']['window_size'])

        self.setup_data()

    def setup_data(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, '..', 'data', config['data']['file_path'])
            logger.info(f"Attempting to process data from: {file_path}")
            
            self.n_scats = tk.StringVar(value='all')
            result = process_data(file_path, n_scats=None)
            
            if isinstance(result, tuple) and len(result) == 3:
                self.df, self.scaler, self.street_segments = result
                self.model_data = prepare_model_data(self.df)
                self.setup_models()
                self.create_widgets()
            else:
                raise ValueError(f"Unexpected result from process_data: {result}")
                
        except Exception as e:
            logger.error(f"Error setting up data: {e}", exc_info=True)
            self.show_error_message(f"Failed to load data: {str(e)}\nPlease check the console for more information.")

    def setup_models(self):
        self.models = {}
        self.available_scats = {}
        for name in config['training']['models']:
            try:
                self.models[name] = {}
                self.available_scats[name] = set()
                for scats in self.df['SCATS Number'].unique():
                    model_path = os.path.join('model', 'trained', f'{name}_{scats}.h5')
                    if os.path.exists(model_path):
                        self.models[name][scats] = load_model(model_path)
                        self.available_scats[name].add(scats)
                    else:
                        logger.warning(f"Model file not found: {model_path}")
            except Exception as e:
                logger.error(f"Error loading {name} models: {e}")

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.create_prediction_tab()
        self.create_route_guidance_tab()
        self.create_map_tab()

    def create_prediction_tab(self):
        pred_frame = ttk.Frame(self.notebook)
        self.notebook.add(pred_frame, text="Traffic Prediction")

        input_frame = ttk.LabelFrame(pred_frame, text="Prediction Inputs")
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(input_frame, text="Select Model:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(input_frame, textvariable=self.model_var, values=list(self.models.keys()))
        self.model_dropdown.grid(row=0, column=1, padx=5, pady=5)
        self.model_dropdown.set(list(self.models.keys())[0])

        ttk.Label(input_frame, text="Select Street:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.street_var = tk.StringVar()
        self.street_dropdown = ttk.Combobox(input_frame, textvariable=self.street_var)
        self.street_dropdown.grid(row=1, column=1, padx=5, pady=5)
        
        self.model_var.trace('w', self.update_street_options)
        self.update_street_options()

        ttk.Label(input_frame, text="Select Date:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.date_var = tk.StringVar()
        self.date_dropdown = ttk.Combobox(input_frame, textvariable=self.date_var, values=sorted(self.df['Date'].dt.date.unique().astype(str)))
        self.date_dropdown.grid(row=2, column=1, padx=5, pady=5)
        self.date_dropdown.set(self.df['Date'].dt.date.unique()[0])

        ttk.Label(input_frame, text="Select Time:").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.time_var = tk.StringVar()
        times = [f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in range(0, 60, 15)]
        self.time_dropdown = ttk.Combobox(input_frame, textvariable=self.time_var, values=times)
        self.time_dropdown.grid(row=3, column=1, padx=5, pady=5)
        self.time_dropdown.set(times[0])

        ttk.Label(input_frame, text="Number of SCATS:").grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.n_scats_dropdown = ttk.Combobox(input_frame, textvariable=self.n_scats, values=config['data']['n_scats_options'])
        self.n_scats_dropdown.grid(row=4, column=1, padx=5, pady=5)

        ttk.Button(input_frame, text="Predict", command=self.predict).grid(row=5, column=0, columnspan=2, pady=10)

        result_frame = ttk.LabelFrame(pred_frame, text="Prediction Results")
        result_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.result_text = tk.Text(result_frame, height=5, width=40, wrap=tk.WORD, font=('Arial', 10))
        self.result_text.pack(pady=5, padx=5, fill=tk.X)

        self.graph_frame = ttk.Frame(result_frame)
        self.graph_frame.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

    def create_route_guidance_tab(self):
        route_frame = ttk.Frame(self.notebook)
        self.notebook.add(route_frame, text="Route Guidance")

        input_frame = ttk.LabelFrame(route_frame, text="Route Inputs")
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(input_frame, text="Origin SCATS:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.origin_var = tk.StringVar()
        self.origin_dropdown = ttk.Combobox(input_frame, textvariable=self.origin_var, values=sorted(self.df['SCATS Number'].unique()))
        self.origin_dropdown.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Destination SCATS:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.destination_var = tk.StringVar()
        self.destination_dropdown = ttk.Combobox(input_frame, textvariable=self.destination_var, values=sorted(self.df['SCATS Number'].unique()))
        self.destination_dropdown.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(input_frame, text="Find Routes", command=self.find_routes).grid(row=2, column=0, columnspan=2, pady=10)

        result_frame = ttk.LabelFrame(route_frame, text="Route Results")
        result_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.route_text = tk.Text(result_frame, height=20, width=60, wrap=tk.WORD, font=('Arial', 10))
        self.route_text.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

    def create_map_tab(self):
        map_frame = ttk.Frame(self.notebook)
        self.notebook.add(map_frame, text="Traffic Map")

        ttk.Button(map_frame, text="Show Traffic Map", command=self.show_map).pack(pady=10)
        self.map_frame = ttk.Frame(map_frame)
        self.map_frame.pack(fill=tk.BOTH, expand=True)

    def predict(self):
        try:
            model_name = self.model_var.get().lower()
            street = self.street_var.get()
            date = self.date_var.get()
            time = self.time_var.get()
            
            street_data = self.df[self.df['Street'] == street]
            available_scats = street_data[street_data['SCATS Number'].isin(self.available_scats[model_name])]
            
            if available_scats.empty:
                raise ValueError(f"No trained model found for {model_name} on the selected street")
            
            scats_number = available_scats['SCATS Number'].iloc[0]
            model = self.models[model_name][scats_number]
            
            X_test = self.model_data[scats_number]['X_test']
            y_test = self.model_data[scats_number]['y_test']
            
            selected_datetime = pd.to_datetime(f"{date} {time}")
            time_index = available_scats[available_scats['DateTime'] == selected_datetime].index[0]
            
            if model_name in ['lstm', 'gru', 'bilstm', 'cnnlstm']:
                input_data = np.reshape(X_test[time_index], (1, X_test.shape[1], 1))
            else:  # SAES
                input_data = np.reshape(X_test[time_index], (1, X_test.shape[1]))

            prediction = model.predict(input_data)
            prediction = self.scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
            
            actual_traffic = self.scaler.inverse_transform(y_test[time_index].reshape(-1, 1))[0][0]
            
            result_text = format_prediction_result(street, date, time, prediction, actual_traffic)
            
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, result_text)
            self.result_text.config(state=tk.DISABLED)
            
            self.plot_prediction(street, date, time, prediction, actual_traffic)
            
        except Exception as e:
            logger.exception(f"Prediction error: {str(e)}")
            messagebox.showerror("Prediction Error", str(e))

    def plot_prediction(self, street, date, time, prediction, actual_traffic):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(10, 6))
        street_data = self.df[(self.df['Street'] == street) & (self.df['Date'].dt.date == pd.to_datetime(date).date())]
        
        plot_prediction(ax, street_data, prediction, actual_traffic, time)
        highlight_time_range(ax, "07:00", "10:00", "Morning Rush")
        highlight_time_range(ax, "16:00", "19:00", "Evening Rush")

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def find_routes(self):
        try:
            origin = self.origin_var.get()
            destination = self.destination_var.get()

            if not origin or not destination:
                messagebox.showerror("Input Error", "Please select both origin and destination SCATS numbers.")
                return

            routes = route_guidance(self.df, self.street_segments, origin, destination)

            self.route_text.config(state=tk.NORMAL)
            self.route_text.delete('1.0', tk.END)
            self.route_text.insert(tk.END, format_route_result(routes))
            self.route_text.config(state=tk.DISABLED)

        except Exception as e:
            logger.exception(f"Route finding error: {str(e)}")
            messagebox.showerror("Route Finding Error", f"An error occurred: {str(e)}")

    def show_map(self):
        try:
            progress_window = tk.Toplevel(self.master)
            progress_window.title("Creating Map")
            progress_label = ttk.Label(progress_window, text="Generating traffic map...")
            progress_label.pack(pady=10)
            progress_bar = ttk.Progressbar(progress_window, length=300, mode="indeterminate")
            progress_bar.pack(pady=10)
            progress_bar.start()

            m = create_traffic_map(self.df, self.street_segments)

            data = io.BytesIO()
            m.save(data, close_file=False)
            img = Image.open(data)
            photo = ImageTk.PhotoImage(img)

            for widget in self.map_frame.winfo_children():
                widget.destroy()

            label = ttk.Label(self.map_frame, image=photo)
            label.image = photo
            label.pack(fill=tk.BOTH, expand=True)

            progress_window.destroy()

        except Exception as e:
            logger.exception(f"Error creating map: {str(e)}")
            messagebox.showerror("Map Creation Error", f"An error occurred while creating the map: {str(e)}")
        finally:
            if 'progress_window' in locals():
                progress_window.destroy()
    
    def update_street_options(self, *args):
        model_name = self.model_var.get().lower()
        available_scats = self.available_scats.get(model_name, set())
        available_streets = sorted(self.df[self.df['SCATS Number'].isin(available_scats)]['Street'].unique())
        
        self.street_dropdown['values'] = available_streets
        if available_streets:
            self.street_var.set(available_streets[0])
        else:
            self.street_var.set('')

    def show_error_message(self, message):
        error_frame = ttk.Frame(self.master)
        error_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        error_label = ttk.Label(error_frame, text=message, foreground="red", wraplength=600, justify=tk.CENTER)
        error_label.pack(expand=True)

        ttk.Button(error_frame, text="Exit", command=self.master.quit).pack(pady=10)

def main():
    root = tk.Tk()
    gui = TFPSGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()