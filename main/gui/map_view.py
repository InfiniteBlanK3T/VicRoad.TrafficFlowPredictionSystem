import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from keras.models import load_model
from data.dataSCATSMap import process_data, prepare_model_data, create_traffic_map
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
import folium
from folium.plugins import MarkerCluster
import webbrowser
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TFPSGUI:
    def __init__(self, master):
        self.master = master
        master.title("Traffic Flow Prediction System")
        master.geometry("1200x800")

        if self.setup_data():
            self.setup_models()
            self.create_widgets()
        else:
            self.show_error_message("Failed to load data. Please check the console for more information.")

    def setup_data(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'data', 'Scats-Data-October-2006-Bundoora.csv')
        print(file_path)
        self.df, self.scaler, self.street_segments = process_data(file_path)
        if self.df is None or self.scaler is None or self.street_segments is None:
            return False
        self.model_data = prepare_model_data(self.df)
        return True

    def show_error_message(self, message):
        error_label = ttk.Label(self.master, text=message, foreground="red")
        error_label.pack(pady=20)
        
    def setup_models(self):
        self.models = {}
        model_names = ['LSTM', 'GRU', 'BiLSTM', 'CNN-LSTM', 'SAES']
        for name in model_names:
            try:
                model_path = os.path.join('model', 'ScatsData-Bundoora', f'{name.lower().replace("-", "")}.h5')
                self.models[name] = load_model(model_path)
            except Exception as e:
                logger.error(f"Error loading {name} model: {e}")

    def create_widgets(self):
        # Prediction Frame
        pred_frame = ttk.LabelFrame(self.master, text="Traffic Volume Prediction")
        pred_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(pred_frame, text="Select Model:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(pred_frame, textvariable=self.model_var, values=list(self.models.keys()))
        self.model_dropdown.grid(row=0, column=1, padx=5, pady=5)
        self.model_dropdown.set("LSTM")

        ttk.Label(pred_frame, text="Select Street:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.street_var = tk.StringVar()
        self.street_dropdown = ttk.Combobox(pred_frame, textvariable=self.street_var, values=sorted(self.df['Street'].unique()))
        self.street_dropdown.grid(row=1, column=1, padx=5, pady=5)
        self.street_dropdown.set(self.df['Street'].unique()[0])

        ttk.Label(pred_frame, text="Select Date:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.date_var = tk.StringVar()
        self.date_dropdown = ttk.Combobox(pred_frame, textvariable=self.date_var, values=sorted(self.df['Date'].dt.date.unique().astype(str)))
        self.date_dropdown.grid(row=2, column=1, padx=5, pady=5)
        self.date_dropdown.set(self.df['Date'].dt.date.unique()[0])

        ttk.Label(pred_frame, text="Select Time:").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.time_var = tk.StringVar()
        times = [f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in range(0, 60, 15)]
        self.time_dropdown = ttk.Combobox(pred_frame, textvariable=self.time_var, values=times)
        self.time_dropdown.grid(row=3, column=1, padx=5, pady=5)
        self.time_dropdown.set(times[0])

        ttk.Button(pred_frame, text="Predict", command=self.predict).grid(row=4, column=0, columnspan=2, pady=10)

        self.result_text = tk.Text(pred_frame, height=5, width=40, wrap=tk.WORD, font=('Arial', 10))
        self.result_text.grid(row=5, column=0, columnspan=2, pady=5, padx=5)
        self.result_text.config(state=tk.DISABLED)

        # Map Frame
        map_frame = ttk.LabelFrame(self.master, text="Traffic Map")
        map_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")

        ttk.Button(map_frame, text="Show Map", command=self.show_map).pack(pady=10)

        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_rowconfigure(1, weight=1)

    def predict(self):
        try:
            model_name = self.model_var.get()
            street = self.street_var.get()
            date = self.date_var.get()
            time = self.time_var.get()
            
            model = self.models[model_name]
            
            # Get the correct input data for the selected street
            street_data = self.df[self.df['Street'] == street]
            scats_number = street_data['SCATS Number'].iloc[0]  # Use the first SCATS number for this street
            X_test = self.model_data[scats_number]['X_test']
            y_test = self.model_data[scats_number]['y_test']
            
            # Find the index for the selected date and time
            selected_datetime = pd.to_datetime(f"{date} {time}")
            time_index = street_data[street_data['DateTime'] == selected_datetime].index[0]
            
            if model_name in ['LSTM', 'GRU', 'BiLSTM', 'CNN-LSTM']:
                input_data = np.reshape(X_test[time_index], (1, X_test.shape[1], 1))
            else:  # SAES
                input_data = np.reshape(X_test[time_index], (1, X_test.shape[1]))

            prediction = model.predict(input_data)
            prediction = self.scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
            
            actual_traffic = self.scaler.inverse_transform(y_test[time_index].reshape(-1, 1))[0][0]
            
            result_text = (
                f"Street: {street}\n"
                f"Date: {date}\n"
                f"Time: {time}\n"
                f"{'Predicted:':>10} {prediction:.2f} vehicles\n"
                f"{'Actual:':>10} {actual_traffic:.2f} vehicles\n"
                f"{'Difference:':>10} {abs(prediction - actual_traffic):.2f} vehicles"
            )
            
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, result_text)
            self.result_text.config(state=tk.DISABLED)
            
        except Exception as e:
            logger.exception(f"Prediction error: {str(e)}")
            messagebox.showerror("Prediction Error", f"An error occurred: {str(e)}")

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

            map_path = 'traffic_map.html'
            m.save(map_path)

            progress_window.destroy()

            webbrowser.open('file://' + os.path.realpath(map_path))
        except Exception as e:
            logger.exception(f"Error creating map: {str(e)}")
            messagebox.showerror("Map Creation Error", f"An error occurred while creating the map: {str(e)}")
        finally:
            if 'progress_window' in locals():
                progress_window.destroy()

def main():
    root = tk.Tk()
    gui = TFPSGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()