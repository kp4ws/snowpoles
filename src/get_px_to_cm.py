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

#3 stakes per image
STAKES = ["1", "2", "3"]

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
    processed_cameras = set()  # Track which cameras we've already processed
    meta_camera_ids = []
    meta_stake_ids = []
    full_pole_length_pxs =[]
    pole_length_cms = []
    conversions = []
    heights = []
    widths = []

    for j, file in tqdm.tqdm(enumerate(dir)):
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
        
        #TODO: Skipping 10cm calibration (possibly add this back in later if things aren't working)
        # figure = plt.figure(figsize=(20, 10), num=Path(file).name)
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # ax = plt.gca()
        # enable_scroll_zoom_and_pan(ax)

        # plt.title(f"STAKE #{stake}: label top and then bottom of 10cm section \n Click ANYWHERE to confirm | BACKSPACE to undo | RIGHT-CLICK drag | SCROLL zoom", fontweight="bold")
        # points = plt.ginput(3, timeout=0, mouse_pop=2)
        # top_10, bottom_10 = points[0], points[1]
        # plt.close()

        figure = plt.figure(figsize=(20, 10), num=Path(file).name)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        ax = plt.gca()
        enable_scroll_zoom_and_pan(ax)
        # TODO: Add in functionality to skip bad images

        plt.title(f"{cameraID}: click top/bottom for stake 1, then stake 2, then stake 3 \n Click ANYWHERE to confirm | BACKSPACE to undo | RIGHT-CLICK drag | SCROLL zoom.", fontweight="bold")
        points = plt.ginput(7, timeout=0, mouse_pop=2)
        plt.close()
        
        for k, stake in enumerate(STAKES):
            top = points[2*k] 
            bottom = points[2*k + 1]

            full_pole_length_px = math.dist((top), (bottom))
            full_pole_length_pxs.append(full_pole_length_px)
            
            #NOTE: Instead of doing calibration step (10 cm), we use recorded measurements from file
            # full_pole_length_cm = (10 / math.dist((top_10), (bottom_10))) *  math.dist((top), (bottom))
            measurement_lookup = get_stake_lengths("CHRL_data/stake_measurements_clean.csv")
            full_pole_length_cm = measurement_lookup[(cameraID, stake)]
            pole_length_cms.append(full_pole_length_cm)

            conversion = full_pole_length_cm / full_pole_length_px 
            conversions.append(conversion)
            
            meta_camera_ids.append(cameraID)
            meta_stake_ids.append(stake)
            
            height, width, channel = img.shape
            heights.append(height)
            widths.append(width)

    metadata = pd.DataFrame(
        {
            "camera_id": meta_camera_ids,
            "stake_id": meta_stake_ids,
            "pole_length_px": full_pole_length_pxs,
            "pole_length_cm": pole_length_cms,
            "pixel_cm_conversion": conversions,
            "width": widths,
            "height": heights,
        }
    )
    metadata.to_csv(f"{args.path}/pole_metadata.csv", index=False)


if __name__ == "__main__":
    main()