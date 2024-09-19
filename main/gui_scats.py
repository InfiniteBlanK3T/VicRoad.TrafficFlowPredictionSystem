import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from keras.models import load_model
from data.dataSCATS import process_data
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta

class TFPSGUI:
    def __init__(self, master):
        self.master = master
        master.title("Traffic Flow Prediction System")
        master.geometry("1000x700")

        self.setup_models()
        self.setup_data()
        self.create_widgets()

    def setup_models(self):
        self.models = {}
        model_names = ['LSTM', 'GRU', 'BiLSTM', 'CNN-LSTM', 'SAES']
        for name in model_names:
            try:
                model_path = os.path.join('model', 'ScatsData-Bundoora', f'{name.lower().replace("-", "")}.h5')
                self.models[name] = load_model(model_path)
            except Exception as e:
                print(f"Error loading {name} model: {e}")

    def setup_data(self):
        file_path = os.path.join('data', 'Scats-Data-October-2006-Bundoora.csv')
        self.lag = 12
        _, _, self.X_test, _, self.scaler = process_data(file_path, self.lag)

    def create_widgets(self):
        # Prediction Frame
        pred_frame = ttk.LabelFrame(self.master, text="Traffic Volume Prediction")
        pred_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(pred_frame, text="Select Model:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(pred_frame, textvariable=self.model_var, values=list(self.models.keys()))
        self.model_dropdown.grid(row=0, column=1, padx=5, pady=5)
        self.model_dropdown.set("LSTM")

        ttk.Label(pred_frame, text="Select Time:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.time_var = tk.StringVar()
        times = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 15, 30, 45]]
        self.time_dropdown = ttk.Combobox(pred_frame, textvariable=self.time_var, values=times)
        self.time_dropdown.grid(row=1, column=1, padx=5, pady=5)
        self.time_dropdown.set("08:00")

        ttk.Button(pred_frame, text="Predict", command=self.predict).grid(row=2, column=0, columnspan=2, pady=10)
        
        self.result_var = tk.StringVar()
        ttk.Label(pred_frame, textvariable=self.result_var).grid(row=3, column=0, columnspan=2, pady=5)
        
        
        # Info Frame
        info_frame = ttk.LabelFrame(self.master, text="Information")
        info_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        ttk.Label(info_frame, text="Traffic volume represents the number of vehicles\npassing through the SCATS site in a 15-minute interval.\nData shown is for a typical weekday.").grid(padx=5, pady=5)
        
        ttk.Label(info_frame, text="--Hit Thomas@BlanK3T up for bugs!-- :)").grid(row=4, column=0, columnspan=2, pady=5)

        # Graph Frame
        self.graph_frame = ttk.LabelFrame(self.master, text="Traffic Volume Graph")
        self.graph_frame.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")

        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_rowconfigure(2, weight=1)

    def predict(self):
        try:
            model_name = self.model_var.get()
            time_str = self.time_var.get()
            time_obj = datetime.strptime(time_str, "%H:%M")
            time_step = time_obj.hour * 4 + time_obj.minute // 15
            
            model = self.models[model_name]
            
            if model_name in ['LSTM', 'GRU', 'BiLSTM', 'CNN-LSTM']:
                input_data = np.reshape(self.X_test[time_step], (1, self.lag, 1))
            else:  # SAES
                input_data = np.reshape(self.X_test[time_step], (1, self.lag))

            prediction = model.predict(input_data)
            prediction = self.scaler.inverse_transform(prediction)[0][0]

            self.result_var.set(f"Predicted traffic volume at {time_str}: {prediction:.2f} vehicles")
            self.plot_prediction(time_step, prediction)
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred: {str(e)}")

    def plot_prediction(self, time_step, prediction):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(8, 6))
        times = [f"{i//4:02d}:{(i%4)*15:02d}" for i in range(96)]
        actual_data = self.scaler.inverse_transform(self.X_test[:96, -1].reshape(-1, 1)).flatten()
        
        ax.plot(times, actual_data, label='Actual', color='blue', alpha=0.7)
        ax.scatter(times[time_step], prediction, color='red', s=100, zorder=5, label='Prediction')
        
        ax.set_xlabel('Time of Day')
        ax.set_ylabel('Traffic Volume (vehicles per 15 min)')
        ax.set_title('24-Hour Traffic Volume Prediction')
        ax.legend()
        
        # Improve x-axis readability
        ax.set_xticks([f"{h:02d}:00" for h in range(0, 24, 3)])
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], rotation=45, ha='right')
        
        # Add vertical gridlines
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Highlight time ranges
        self.highlight_time_range(ax, "07:00", "10:00", "Morning Rush")
        self.highlight_time_range(ax, "16:00", "19:00", "Evening Rush")

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def highlight_time_range(self, ax, start, end, label):
        start_idx = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 15, 30, 45]].index(start)
        end_idx = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 15, 30, 45]].index(end)
        ax.axvspan(start, end, alpha=0.2, color='yellow')
        ax.text(start, ax.get_ylim()[1], label, ha='left', va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

def main():
    root = tk.Tk()
    gui = TFPSGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()