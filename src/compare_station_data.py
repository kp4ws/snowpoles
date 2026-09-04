import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from scipy.stats import linregress
import matplotlib.pyplot as plt
from arg_parser import ArgumentParser
import os

def main():
    args = ArgumentParser("Evaluate model on the train/val images")
    
    #NOTE: This is currently ONLY comparing CC1 prediction with CC1 weather station data.
    #Future iteration would involve looping through each camera site and running it against corresponding weather station
    #Since not all camera sites have nearby weather stations, a paramter could be added into the configuration file to determine if site is by a station or not.
    results_path = f"{args.models_output}/predictions/results.csv"
    station_path = f"{args.station_path}/CC1_merged.csv"

    if not os.path.exists(station_path):
        print(f"Warning: station data path doesn't exist")
    
    target_camera = "CC1"

    try:
        results_df = pd.read_csv(results_path, parse_dates=['datetime'])
        station_df = pd.read_csv(station_path, parse_dates=['datetime'])
        # Ensure datetime columns are proper pandas datetime objects and aligned
        results_df['datetime'] = pd.to_datetime(results_df['datetime']).dt.round('min')
        station_df['datetime'] = pd.to_datetime(station_df['datetime']).dt.round('min')

    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    #Filter for specific camera site
    camera_df = results_df[results_df["camera_id"] == target_camera].copy()

    if camera_df.empty:
        print(f"No prediction data found for camera {target_camera}")
        return

    #Align timestamps of each dataframe
    # Align timestamps and strip timezones
    camera_df["datetime"] = pd.to_datetime(camera_df["datetime"], errors="coerce").dt.tz_localize(None)
    station_df["datetime"] = pd.to_datetime(station_df["datetime"], errors="coerce").dt.tz_localize(None)
    
    camera_df = camera_df.dropna(subset=["datetime"]).sort_values("datetime")
    station_df = station_df.dropna(subset=["datetime"]).sort_values("datetime")

    # Use merge_asof to find the closest station reading within a 2-hour window
    merged_df = pd.merge_asof(
        camera_df, 
        station_df, 
        on="datetime", 
        direction="nearest",
        tolerance=pd.Timedelta(hours=2) 
    )

    # Drop any camera rows that didn't find a station reading within that 2-hour window
    merged_df = merged_df.dropna(subset=["snow_depth_m"])

    if merged_df.empty:
        print("Error: No overlapping timestamps found between the camera and the station data.")
        return

    #Extract arrays for each of the datasets from the merged dataframe
    try:
        pred_depth = merged_df["s1_snow_depth"] #NOTE: Currently using snow stake 1 for comparison (no specific reason for choosing 2, just appears not to shift much).
        true_depth = merged_df["snow_depth_m"]
        true_depth = true_depth * 100  # convert from meters to centimeters
    except KeyError as e:
        print(f"Missing expected column in CSV: {e}")
        return

    #Remove NaNs (to prevent math errors) and drop predictions that fall below an impossible threshold (e.g., -5 cm)
    valid_idx = ~np.isnan(pred_depth) & ~np.isnan(true_depth) & (pred_depth > -5)
    true_depth = true_depth[valid_idx]
    pred_depth = pred_depth[valid_idx]

    #Calc stats
    slope, intercept, r_value, p_value, std_err = linregress(true_depth, pred_depth)
    r2 = r2_score(true_depth, pred_depth)
    mae = mean_absolute_error(true_depth, pred_depth)
    rmse = root_mean_squared_error(true_depth, pred_depth)

    print(f"\n--- Stats for {target_camera} (Pole 1) ---")
    print(f"Data points compared: {len(true_depth)}")
    print(f"R^2: {r2:.4f}")
    print(f"MAE: {mae:.2f} cm")
    print(f"RMSE: {rmse:.2f} cm")
    print(f"Regression Line: y= {slope:.2f}x + {intercept:.2f}\n")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Time Series Comparison
    ax1.plot(merged_df["datetime"][valid_idx], true_depth, label="Station (Ground Truth)", color="blue", marker="o", markersize=4)
    ax1.plot(merged_df["datetime"][valid_idx], pred_depth, label="Camera Prediction", color="red", marker="x", markersize=4)
    ax1.set_title(f"{target_camera} Snow Depth Over Time")
    ax1.set_ylabel("Snow Depth (cm)")
    ax1.set_xlabel("Date")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=45)

    # Plot 2: Scatter Plot & Regression Line
    ax2.scatter(true_depth, pred_depth, color="purple", alpha=0.6)
    x_vals = np.array(ax2.get_xlim())
    y_vals = intercept + slope * x_vals
    ax2.plot(x_vals, y_vals, "--", color="black", label=f"Best Fit (R^2={r2:.2f})")

    # Ideal 1:1 line (where predictions perfectly match truth)
    ax2.plot(x_vals, x_vals, "g:", label="Perfect 1:1 match")

    ax2.set_title("Predicted vs Actual Depth")
    ax2.set_xlabel("Station Depth (cm)")
    ax2.set_ylabel("Predicted Depth (cm)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()