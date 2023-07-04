import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import tkinter as tk
from tkinter import ttk

class BitcoinPriceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bitcoin Price Data")
        self.root.geometry("400x400")
        
        # Create start and end date entry boxes
        self.start_label = ttk.Label(self.root, text="Start Date (YYYY-MM-DD):")
        self.start_label.pack(pady=(10,5))
        self.start_entry = ttk.Entry(self.root)
        self.start_entry.pack()
        
        self.end_label = ttk.Label(self.root, text="End Date (YYYY-MM-DD):")
        self.end_label.pack(pady=(10,5))
        self.end_entry = ttk.Entry(self.root)
        self.end_entry.pack()
        
        # Create a button to get the historical data
        self.get_data_button = ttk.Button(self.root, text="Get Data", command=self.get_historical_data)
        self.get_data_button.pack(pady=10)
        
        # Create a figure to display the plot
        self.fig = plt.figure(figsize=(8,6))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price (USD)")
        self.canvas = plt.gcf().canvas
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
    def get_historical_data(self):
        # Get the start and end dates from the entry boxes
        start_date = self.start_entry.get()
        end_date = self.end_entry.get()
        
        # Set the API endpoint and parameters
        url = "https://api.pro.coinbase.com/products/BTC-USD/candles"
        params = {
            "granularity": 86400,  # 1 day interval
            "start": f"{start_date}T00:00:00Z",  # Start date for historical data
            "end": f"{end_date}T00:00:00Z"  # End date for historical data
        }

        # Send the API request and store the response as a pandas dataframe
        response = requests.get(url, params=params)
        df = pd.DataFrame(response.json(), columns=["time", "low", "high", "open", "close", "volume"])

        # Convert the timestamp to a readable datetime format
        df["time"] = pd.to_datetime(df["time"], unit="s")
        
        # Plot the closing prices
        self.ax.clear()
        self.ax.plot(df["time"], df["close"])
        self.fig.autofmt_xdate()
        self.canvas.draw()
        
# Create the GUI window
root = tk.Tk()
BitcoinPriceGUI(root)
root.mainloop()
