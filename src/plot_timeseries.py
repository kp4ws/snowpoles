"""
Author: Kent Pawson (2026)

Used to plot the snow depth time series from the predict.py results

py src/plot_timeseries.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from pathlib import Path
from arg_parser import ArgumentParser

def main():
    args = ArgumentParser(description="Plot snow depth time series from results.csv")
    
    os.makedirs(args.path, exist_ok=True)

    print(f"Loading data from ...") #TODO
    df = pd.read_csv(f"{args.models_output}/predictions/results.csv")
    
    df["datetime"] = pd.to_datetime(df['datetime'], format="%m/%d/%Y %H:%M")
    df = df.sort_values("datetime")

    depth_cols = [col for col in df.columns if col.endswith("_snow_depth")]

    if not depth_cols:
        print("Error: no snow depth columns found in the CSV")
        return
    
    cameras = df["camera_id"].unique()

    for cam in cameras:
        cam_df = df[df["camera_id"] == cam]

        plt.figure(figsize=(12, 6))

        #plot a line for each active pole at the current camera site
        for col in depth_cols:
            pole_data = cam_df[["datetime", col]].dropna()

            if not pole_data.empty:
                pole_id = col.split("_")[0].upper()
                plt.plot(
                    pole_data["datetime"],
                    pole_data[col],
                    marker="o", #TODO dots might be too big
                    linestyle="-",
                    markersize=4,
                    label=f"Pole {pole_id}"
                )

        
        plt.title(f"Snow Depth Time Series: Camera {cam}", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Snow Depth (cm)", fontsize=12)

        #Format x-axis to show readable dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        save_path = Path(args.path) / f"{cam}_timeseries.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved plot for {cam} to {save_path}")

if __name__ == "__main__":
    main()