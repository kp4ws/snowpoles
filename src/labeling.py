"""
written by Catherine M. Breen 
cbreen@uw.edu 

Use of our keypoint detection model currently requires ~10 images per camera. We provide a labeling script below that when pointed 
at a camera directory (i.e., data > cam1 or data > cam2, etc), walks the user through labeling every 10th image and saves as labels.csv in a specified direrctory. 

We estimate it will take about 5 imgs/min or about 300 imgs per hour. 

x1,y1 = top 
x2,y2 = bottom

The labels.csv file can then be directly pointed at train.py for fine-tuning. The user can then run predict.py to extract the snow depth.

example run 

python src/labeling.py --datapath "/path/to/nontrained/data" --pole_length "304.8" --subset_to_label "2"
python src/labeling.py --datapath "/Users/cmbreen/Documents/FDLTCC/FF_2024" --subset_to_label "10"

python src/labeling.py --datapath "/Users/cmbreen/Documents/FDLTCC/summer_2025/FF_2024" --subset_to_label "10"



"""

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

def main():
    # Argument parser for command-line arguments:
    args = ArgumentParser("Manually label images for training", "label")

    metadata_path = Path(args.path) / "pole_metadata.csv"
    meta_df = pd.read_csv(metadata_path)

    ## customized data
    #pole_length = np.float64(args.pole_length)
    subset_to_label = np.int16(args.subset_to_label)

    ## load labels.csv
    labels_path = Path(args.path) / "labels.csv"

    required_cols = ["filename", "camera_id", "datetime"]

    for p in range(args.number_of_poles):
        poleId = p + 1
        required_cols.extend([
            f"s{poleId}_x1", f"s{poleId}_y1",
            f"s{poleId}_x2", f"s{poleId}_y2",
            f"s{poleId}_pixel_length", f"s{poleId}_snow_depth"
        ])

    if labels_path.exists():
        try:
            df_existing = pd.read_csv(labels_path, skip_blank_lines=True)
            if not all(col in df_existing.columns for col in required_cols):
                raise ValueError("Missing required columns")
        except Exception:
            print("labels.csv is corrupted or does not exist, creating...")
            df_existing = pd.DataFrame(columns=required_cols)
            df_existing.to_csv(labels_path, index=False)
    else:
        print("labels.csv is corrupted or does not exist, creating...")
        df_existing = pd.DataFrame(columns=required_cols)
        df_existing.to_csv(labels_path, index=False) 

    #Tracks which images have already been labeled
    already_labeled = set(zip(
        df_existing["filename"].astype(str),
        df_existing["camera_id"].astype(str)
    ))

    ### loop to label every nth photo!
    input_images = Path(args.path)
    cam_dirs = [item for item in input_images.iterdir() if item.is_dir()]

    for cam_dir in cam_dirs:
        camera_id = cam_dir.name

        #NOTE: This filters out all cameras except CTRL1
        if camera_id != "CTRL1":
            continue

        try:
            cam_meta = meta_df[meta_df["camera_id"] == camera_id].iloc[0]
        except IndexError:
            print(f"Warning: No calibration data found in pole_metadata.csv for camera: {camera_id}. Skipping directory.")
            continue

        pole_length_cm_lookup = {}
        conversion_lookup = {}

        for p in range(args.number_of_poles):
            poleId = p + 1
            pole_length_cm_lookup[poleId] = float(cam_meta[f"s{poleId}_pole_length_cm"])
            conversion_lookup[poleId] = float(cam_meta[f"s{poleId}_pixel_cm_conversion"])

        files = sorted(cam_dir.glob("*.JPG"))
        for image_index, file_path in enumerate(files):
            if image_index % subset_to_label != 0:
                continue
            
            #Avoid labeling images that have already been labeled
            if(file_path.name, camera_id) in already_labeled:
                continue

            #TODO: Delete later
            # snow_free_image = "CTRL1_20260504_10.jpg"
            # snow_image = "CTRL1_20260226_11.jpg"
            # if(file_path.name != snow_free_image and file_path.name != snow_image):
            #     continue
            
            img = cv2.imread(str(file_path))
            figure = plt.figure(figsize=(20, 10), num=file_path.name)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax = plt.gca()
            enable_scroll_zoom_and_pan(ax)

            expected_clicks = 2 * args.number_of_poles + 1
            plt.title("label top and then bottom of each pole (left to right) \n Click ANYWHERE to confirm | BACKSPACE to undo | RIGHT-CLICK drag | SCROLL zoom | ENTER to skip.", fontweight="bold")
            plt.tight_layout()
            points = plt.ginput(expected_clicks, timeout=0, mouse_pop=2)
            plt.close()

            if len(points) < expected_clicks:
                print(f"Skipping {file_path.name} (Not enough points collected / User skipped)")
                continue  # This jumps to the next file in your loop
            
            creation_time = os.path.getmtime(file_path)
            formatted_datetime = datetime.datetime.fromtimestamp(creation_time).strftime("%m/%d/%Y %H:%M")

            new_row = {
                "filename": file_path.name,
                "camera_id": camera_id,
                "datetime": formatted_datetime,
            }

            for j in range(args.number_of_poles):
                poleId = j+1
                top = points[2 * j]
                bottom = points[2 * j + 1]
                pixel_length = math.dist(top, bottom)

                #snow depth calculated by subtracting the known pole length by our pixel length (where we clicked in the image) * our conversion (previously calculated)
                #In snow free images, snow depth should be ~0, which can be used as ground truth.
                #In snowy images, our bottom click position should be at the snow interface, and the snow depth value should be higher (depending on amount of snow).
                snow_depth = pole_length_cm_lookup[poleId] - (pixel_length * conversion_lookup[poleId])

                new_row[f"s{poleId}_x1"] = top[0]
                new_row[f"s{poleId}_y1"] = top[1]
                new_row[f"s{poleId}_x2"] = bottom[0]
                new_row[f"s{poleId}_y2"] = bottom[1]
                new_row[f"s{poleId}_pixel_length"] = pixel_length
                new_row[f"s{poleId}_snow_depth"] = snow_depth

            pd.DataFrame([new_row])[required_cols].to_csv(labels_path, mode="a", header=False, index=False)
            print(f"Successfully saved wide row metrics for: {file_path.name}")

if __name__ == "__main__":
    main()