"""
Author: Kent Pawson (2026)

Generates pixel to cm conversion metadata for each of the active camera sites/poles.

py src/px_cm_conversion.py
"""

import cv2
import matplotlib.pyplot as plt
import math
import pandas as pd
from pathlib import Path
from utils import enable_scroll_zoom_and_pan
from arg_parser import ArgumentParser
from config import cameras
import sys

def get_stake_lengths(csv_path: str):
    df = pd.read_csv(csv_path)
    
    stake_lengths = {}
    for _, row in df.iterrows():
        #(cam_id, stake_id)
        key = (row["camera_id"], str(row["stake_id"]))
        stake_lengths[key] = float(row["pole_length_cm"])

    return stake_lengths

def main():
    args = ArgumentParser("Manually label images for training")

    ######## for pole_metdata #######
    all_camera_rows = []

    processed_cameras = set()  # Track which cameras we've already processed
    sites_dir = Path(args.path) / "sites"
    cam_dirs = [x for x in sites_dir.iterdir() if x.is_dir()]

    measurement_lookup = get_stake_lengths("CHRL_data/stake_measurements_clean.csv")

    for cam_dir in cam_dirs:
        camera_id = cam_dir.name
        camera_cfg = cameras.get(camera_id)

        # Skip if cameraID is not in config
        if not camera_cfg:
            print(f"{cam_dir} not found in config, skipping")
            continue

        is_enabled = camera_cfg.get("enabled", True)
        active_poles = camera_cfg.get("active_poles", [])

        # Skip if camera is not enabled or has no active poles
        if not is_enabled or not active_poles:
            print(f"{cam_dir} not active, skipping")
            continue 

        # Skip if we've already processed this camera
        if camera_id in processed_cameras:
            print(f"{cam_dir} already processed, skipping")
            continue

        #Get baseline image for current camera site
        baseline_file = cam_dir / "zz_baseline.jpg"
        if not baseline_file.exists():
            print(f"Baseline image missing for {camera_id}. Please run baseline script first.")
            sys.exit(0)

        processed_cameras.add(camera_id)
        img = cv2.imread(str(baseline_file))
        height, width, channel = img.shape

        #If camera site is marked as upside-down, flip image
        if camera_cfg.get("upside_down", False):
            img = cv2.rotate(img, cv2.ROTATE_180)

        figure = plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        ax = plt.gca()
        enable_scroll_zoom_and_pan(ax)

        expected_clicks = 2 * len(active_poles) + 1

        plt.title(
            f"$\\bf{{{camera_id}}}$  |  $\\bf{{ACTIVE\\ POLES: {active_poles}}}$\n"
            f"1. Click top and bottom of each active pole (left to right)\n"
            f"2. Click anywhere once more to Confirm\n"
            f"[$\\bf{{Backspace:}}$ Undo last click | $\\bf{{Scroll:}}$ Zoom | $\\bf{{Right-Click + Drag:}}$ Pan]",
            fontsize=12, 
            color="#1a1a1a",
            pad=10,
            loc="left", 
            ma="left"
            )
        points = plt.ginput(expected_clicks, timeout=0, mouse_pop=2)
        plt.close()

        if len(points) < expected_clicks:
            print(f"Skipping {camera_id} (Not enough points collected / User skipped)")
            continue

        current_row = {
            "camera_id": camera_id,
            "width": width,
            "height": height,
        }
        
        for j, poleId in enumerate(active_poles):
            top = points[2*j] 
            bottom = points[2*j + 1]
            
            try:    
                full_pole_length_px = math.dist((top), (bottom))
                full_pole_length_cm = measurement_lookup[(camera_id, str(poleId))] #+1 because poleIndex is 0-index
                conversion = full_pole_length_cm / full_pole_length_px 
            except KeyError as e:
                print(e)
                print("Consider setting pole as inactive in configuration")

            current_row[f"s{poleId}_pole_length_px"] = full_pole_length_px
            current_row[f"s{poleId}_pole_length_cm"] = full_pole_length_cm
            current_row[f"s{poleId}_pixel_cm_conversion"] = conversion
        
        all_camera_rows.append(current_row)
            
    metadata_df = pd.DataFrame(all_camera_rows)
    metadata_df.to_csv(f"{args.path}/pole_metadata.csv", index=False)
    print(f"Success: Metadata saved to {args.path}/pole_metadata.csv")

if __name__ == "__main__":
    main()