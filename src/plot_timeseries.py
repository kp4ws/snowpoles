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
from config import timeseries

def main():
    args = ArgumentParser(description="Plot snow depth time series from results.csv")
    os.makedirs(args.path, exist_ok=True)

    station_path = f"{args.station_path}/CC1_merged.csv"
    if not os.path.exists(station_path):
        print(f"Warning: station data path doesn't exist")

    target_camera = "CC1"

    print(f"Loading data from prediction results...")
    df = pd.read_csv(f"{args.models_output}/predictions/results.csv")
    
    df["datetime"] = pd.to_datetime(df['datetime'], format="%m/%d/%Y %H:%M")
    df = df.sort_values("datetime")

    depth_cols = [col for col in df.columns if col.endswith("_snow_depth")]

    if not depth_cols:
        print("Error: no snow depth columns found in the CSV")
        return

    single_average_mode = timeseries.get("single_average_mode", False)
    cameras = df["camera_id"].unique()

    for cam in cameras:
        cam_df = df[df["camera_id"] == cam].copy()
        plt.figure(figsize=(12, 6))

        if single_average_mode:
            cam_df['site_avg_snow_depth'] = cam_df[depth_cols].mean(axis=1)
            avg_data = cam_df.dropna(subset=['site_avg_snow_depth'])

            if not avg_data.empty:
                plt.plot(
                    avg_data['datetime'],
                    avg_data['site_avg_snow_depth'],
                    marker='o',
                    linestyle='-',
                    markersize=4,
                    color="blue",
                    label="Site Average Depth"
                )
        else:
            #plot a line for each active pole at the current camera site
            for col in depth_cols:
                pole_data = cam_df[["datetime", col]].dropna()

                if not pole_data.empty:
                    pole_id = col.split("_")[0].upper()
                    plt.plot(
                        pole_data["datetime"],
                        pole_data[col],
                        marker="o",
                        linestyle="-",
                        markersize=4,
                        label=f"Pole {pole_id}"
                    )

        #If camera = target camera, plot data from weather station to see visual comparison
        if cam == target_camera:
            df_station = pd.read_csv(station_path, parse_dates=['datetime'])

            # Set the index, resample to 1-Hour intervals (filling missing hours with NaN), and reset
            # df_station = df_station.set_index('datetime').resample('1H').mean().reset_index()

            df_discrete = df_station.dropna(subset=['field_snow_depth_m']).copy()
            df_discrete['field_snow_depth_cm'] = df_discrete['field_snow_depth_m'] * 100

            plt.scatter(
                df_discrete['datetime'],
                df_discrete['field_snow_depth_cm'],
                color='black',
                marker='X',
                s=75,
                label="Field Measurements",
                zorder=5
            )

            plt.plot(
                df_station['datetime'],
                df_station['snow_depth_m'] * 100,
                color='purple',
                linestyle='--',
                alpha=0.6,
                label="Station Snow Depth"
            )
        
        plt.title(f"Snow Depth Time Series: Camera {cam}", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Snow Depth (cm)", fontsize=12)

        min_date = cam_df["datetime"].min() - pd.Timedelta(days=2)
        max_date = cam_df["datetime"].max() + pd.Timedelta(days=2)
        plt.xlim(min_date, max_date)

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