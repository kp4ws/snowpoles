import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from scipy.stats import linregress

def main():
    results_path = ""
    station_path = ""

    try:
        results_df = pd.read_csv(results_path)
        station_df = pd.read_csv(station_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    #Filter for specific camera site
    target_camera = "CTRL1" #TODO: Consider adding this into config.toml
    camera_df = results_df[results_df["camera_id"] == target_camera].copy()

    if camera_df.empty:
        print(f"No prediction data found for camera {target_camera}")
        return

    #Merge two datasets on datetime column
    merged_df = pd.merge(camera_df, station_df, on="datetime", how="inner")

    if merged_df.empty:
        print("Error: No overlapping timestamps found between the camera and the station data.")
        return

    #Extract arrays for each of the datasets from the merged dataframe
    try:
        pred_depth = merged_df["s2_snow_depth"] #NOTE: Currently using snow stake 2. TODO: Test out with other snow stakes later on.
        true_depth = merged_df["station_snow_depth"]
    except KeyError as e:
        print(f"Missing expected column in CSV: {e}")
        return

    #Remove NaNs (to prevent math errors)
    valid_idx = ~np.isnan(pred_depth) & ~np.isnan(true_depth)
    true_depth = true_depth[valid_idx]
    pred_depth = pred_depth[valid_idx]

    #Calc stats
    slope, intercept, r_value, p_value, std_err = linregress(true_depth, pred_depth)
    r2 = r2_score(true_depth, pred_depth)
    mae = mean_absolute_error(true_depth, pred_depth)
    rmse = root_mean_squared_error(true_depth, pred_depth)

    print(f"\n--- Stats for {target_camera} (Pole 2) ---")
    print(f"Data points compared: {len(true_depth)}")
    print(f"R^2: {r2:.4f} cm")
    print(f"MAE: {mae:.2f} cm")
    print(f"RMSE: {rmse:.2f} cm")
    print(f"Regression Line: y= {slope:.2f}x + {intercept:.2f}\n")

    #Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    #Plot 1: Time Series Comparison
    ax1.plot(merged_df["datetime"][valid_idx], true_depth, label="Station (Ground Truth)", color="blue", marker="o", markersize=4)
    ax1.plot(merged_df["datetime"][valid_idx], pred_depth, label="Camera Prediction", color="red", marker="x", markersize=4)
    ax1.set_title(f"{target_camera} Snow Depth Over Time")
    ax1.set_ylabel("Snow Depth (cm)")
    ax1.set_xlabel("Date")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=45)

    #Plot 2: Scatter Plot & Regression Line
    ax2.scatter(true_depth, pred_depth, color="purple", alpha=0.6)
    x_vals = np.array(ax2.get_xlim())
    y_vals = intercept + slope * x_vals
    ax2.plot(x_vals, y_vals, "--", color="black", label=f"Best Fit (R^2={r2:.2f})")

    #Ideal 1:1 line (where predictions perfectly match truth)
    ax2.plot(x_vals, x_vals, "g:", label="Perfect 1:1 match")

    ax2.set_title("Predicted vs Actual Depth")
    ax2.set_xlabel("Station Depth (cm)")
    ax2.set_ylabel("Predicted Depth (cm)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()