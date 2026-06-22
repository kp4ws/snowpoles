import cv2
import matplotlib.pyplot as plt
import glob
import argparse
import tqdm
import math
import pandas as pd
import os
import datetime
import numpy as np
from pathlib import Path
import tomli as tomllib
import IPython

from utils import enable_scroll_zoom_and_pan
from arg_parser import ArgumentParser

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

    # dir = glob.glob(f"{args.path}/**/*")  # /*") ## path to data directory
    dir = list(
        Path(args.path).rglob("*.JPG")
    )  # Recursively lists all files and directories
    dir = sorted(dir)
    
    ######## for pole_metdata #######
    metadata = {
        "camera_id": [],
        "width": [],
        "height": [],
    }
    
    processed_cameras = set()  # Track which cameras we've already processed

    for i, file in tqdm.tqdm(enumerate(dir)):
        # Skip if we've already processed this camera
        cameraID = Path(file).parent.name
        if cameraID in processed_cameras:
            continue

        #NOTE: This condition tests for specific site/camera.
        if cameraID != "CTRL1":
            continue

        snow_free_image = "CTRL1_20260504_10.jpg"
        if(file.name != snow_free_image):
            continue

        processed_cameras.add(cameraID)
        img = cv2.imread(str(file))
        height, width, channel = img.shape

        metadata["camera_id"].append(cameraID)
        metadata["width"].append(width)
        metadata["height"].append(height)

        figure = plt.figure(figsize=(20, 10), num=Path(file).name)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        ax = plt.gca()
        enable_scroll_zoom_and_pan(ax)
        # TODO: Add in functionality to skip bad images

        plt.title(
            f"{cameraID}: click top/bottom for poles (left to right) \n Click ANYWHERE to confirm | BACKSPACE to undo | RIGHT-CLICK drag | SCROLL zoom.", fontweight="bold")
        points = plt.ginput(2 * args.number_of_poles + 1, timeout=0, mouse_pop=2)
        plt.close()
        
        #NOTE: Instead of doing calibration step (10 cm), we use recorded measurements from file
        # full_pole_length_cm = (10 / math.dist((top_10), (bottom_10))) *  math.dist((top), (bottom))
        measurement_lookup = get_stake_lengths("CHRL_data/stake_measurements_clean.csv")

        for j in range(args.number_of_poles):
            poleId = j + 1

            top = points[2*j] 
            bottom = points[2*j + 1]

            full_pole_length_px = math.dist((top), (bottom))
            full_pole_length_cm = measurement_lookup[(cameraID, str(poleId))] #+1 because poleIndex is 0-index
            conversion = full_pole_length_cm / full_pole_length_px 
            
            for key, val in [
                (f"s{poleId}_pole_length_px", full_pole_length_px),
                (f"s{poleId}_pole_length_cm", full_pole_length_cm),
                (f"s{poleId}_pixel_cm_conversion", conversion)
            ]:
                if key not in metadata:
                    # Pad with historical Nones if any earlier camera rows were populated
                    metadata[key] = [None] * (len(processed_cameras)- 1)
                metadata[key].append(val)
            
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(f"{args.path}/pole_metadata.csv", index=False)

if __name__ == "__main__":
    main()