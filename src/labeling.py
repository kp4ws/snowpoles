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

#3 stakes per image
STAKES = ["1", "2", "3"]

def main():
    # Argument parser for command-line arguments:
    args = ArgumentParser("Manually label images for training")

    ## labeling data
    camera_ids = []
    stake_ids = []
    filenames = []
    creation_times = []
    topX, topY, bottomX, bottomY = [], [], [], []
    pixel_lengths = []
    snow_depths = []

    metadata_path = Path(args.path) / "pole_metadata.csv"
    meta_df = pd.read_csv(metadata_path)
    meta_df["stake_id"] = meta_df["stake_id"].astype(str)

    meta = meta_df.set_index(["camera_id", "stake_id"])
    pole_length_cm_lookup = meta["pole_length_cm"].to_dict()
    conversion_lookup = meta["pixel_cm_conversion"].to_dict()

    ## customized data
    #pole_length = np.float64(args.pole_length)
    subset_to_label = np.int16(args.subset_to_label)

    ## load labels.csv
    labels_path = Path(args.path) / "labels.csv"
    required_cols = [
        "camera_id",
        "stake_id",
        "filename",
        "datetime",
        "x1",
        "y1",
        "x2",
        "y2",
        "pixel_length",
        "snow_depth",
    ]

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

    camera_ids = df_existing["camera_id"].astype(str).tolist()
    stake_ids = df_existing["stake_id"].astype(str).tolist()
    filenames = df_existing["filename"].astype(str).tolist()
    creation_times = df_existing["datetime"].astype(str).tolist()
    topX = df_existing["x1"].tolist()
    topY = df_existing["y1"].tolist()
    bottomX = df_existing["x2"].tolist()
    bottomY = df_existing["y2"].tolist()
    pixel_lengths = df_existing["pixel_length"].tolist()
    snow_depths = df_existing["snow_depth"].tolist()

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

            plt.title("label top and then bottom of full pole \n Click ANYWHERE to confirm | BACKSPACE to undo | RIGHT-CLICK drag | SCROLL zoom | ENTER to skip.", fontweight="bold")
            plt.tight_layout()
            points = plt.ginput(7, timeout=0, mouse_pop=2)
            plt.close()

            if len(points) < 7:
                print(f"Skipping {file_path.name} (Not enough points collected / User skipped)")
                continue  # This jumps to the next file in your loop
            
            creation_time = os.path.getmtime(file_path)
            formatted_datetime = datetime.datetime.fromtimestamp(creation_time).strftime("%m/%d/%Y %H:%M")

            new_rows = []
            for k, stake in enumerate(STAKES):
                top = points[2 * k]
                bottom = points[2 * k + 1]
                pixel_length = math.dist(top, bottom)

                #snow depth calculated by subtracting the known pole length by our pixel length (where we clicked in the image) * our conversion (previously calculated)
                #In snow free images, snow depth should be ~0, which can be used as ground truth.
                #In snowy images, our bottom click position should be at the snow interface, and the snow depth value should be higher (depending on amount of snow).
                snow_depth = pole_length_cm_lookup[(camera_id, stake)] - (pixel_length * conversion_lookup[(camera_id, stake)])

                camera_ids.append(cam_dir.name)
                stake_ids.append(stake)
                filenames.append(file_path.name)
                creation_times.append(formatted_datetime)
                topX.append(top[0])
                topY.append(top[1])
                bottomX.append(bottom[0])
                bottomY.append(bottom[1])
                pixel_lengths.append(pixel_length)
                snow_depths.append(snow_depth)

                new_rows.append({
                    "camera_id": camera_id,
                    "stake_id": stake,
                    "filename": file_path.name,
                    "datetime": formatted_datetime,
                    "x1": top[0],
                    "y1": top[1],
                    "x2": bottom[0],
                    "y2": bottom[1],
                    "pixel_length": pixel_length,
                    "snow_depth": snow_depth,
                })
            pd.DataFrame(new_rows).to_csv(labels_path, mode="a", header=False, index=False)
    

    # ## simplified table for snow depth conversion later on
    # df = pd.DataFrame(
    #     {
    #         "camera_id": camera_ids,
    #         "stake_id": stake_ids,
    #         "filename": filenames,
    #         "datetime": creation_times,
    #         "x1": topX,
    #         "y1": topY,
    #         "x2": bottomX,
    #         "y2": bottomY,
    #         "pixel_lengths": pixel_lengths,
    #         "snow_depths":snow_depths,
    #     }
    # )

    # df.to_csv(f"{args.path}/labels.csv", index=False)

if __name__ == "__main__":
    main()