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

#3 stakes per image
STAKES = ["1", "2", "3"]

def main():
    # Argument parser for command-line arguments:
    parser = argparse.ArgumentParser(description="Manually label images for training")
    parser.add_argument("--path", help="directory where images are located")
    parser.add_argument(
        "--datapath", help="(deprecated) directory where images are located"
    )
    # parser.add_argument(
    #     "--pole_length", help="length of pole in cm"
    # )
    parser.add_argument(
        "--subset_to_label", help="label every N images"
    )
    parser.add_argument(
        "--no_confirm", required=False, help="skip confirmation", action="store_true"
    )
    args = parser.parse_args()
    args.path = args.datapath

    # Get arguments from config file if they weren't specified
    with open("config.toml", "rb") as configfile:
        config = tomllib.load(configfile)
    if not args.path:
        args.path = config["paths"]["input_images"]
    # if not args.pole_length:
    #     args.pole_length = config["labeling"]["pole_length"]
    if not args.subset_to_label:
        args.subset_to_label = config["labeling"]["subset_to_label"]

    # Confirmation
    if not args.no_confirm:
        print(
            "\n\n# The following options were specified in config.toml or as arguments:\n"
        )
        if (args.path.startswith("/")):
            print(
                "Directory where images are located:\n"
                + str(args.path)
                + "\n"
            )
        else:
            print(
                "Directory where images are located:\n"
                + os.getcwd()
                + "/"
                + str(args.path)
                + "\n"
            )
        #print("Pole length:\n" + str(args.pole_length) + "cm")
        print("\nImages to label:\nEvery", str(args.subset_to_label), "images")
        confirmation = str(input("\n\nIs this OK? (y/n) "))
        if confirmation.lower() != "y":
            if confirmation.lower() == "n":
                print(
                    "\nEdit the config file, located at",
                    os.getcwd()
                    + "/config.toml, to your liking, or edit the command line arguments if they were specified, and then re-run this file.\n",
                )
            else:
                print("Invalid input.\n")
            quit()

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

    ### loop to label every nth photo!
    input_images = Path(args.path)
    cam_dirs = [item for item in input_images.iterdir() if item.is_dir()]
    for cam_dir in cam_dirs:
        camera_id = cam_dir.name
        
        if camera_id != "CTRL1":
            continue

        files = sorted(cam_dir.glob("*.JPG"))
        for image_index, file_path in enumerate(files):
            if image_index % subset_to_label != 0:
                continue
            
            img = cv2.imread(str(file_path))
            figure = plt.figure(figsize=(20, 10), num=file_path.name)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax = plt.gca()
            enable_scroll_zoom_and_pan(ax)

            plt.title("label top and then bottom of full pole \n Click ANYWHERE to confirm | BACKSPACE to undo | RIGHT-CLICK drag | SCROLL zoom.", fontweight="bold")
            plt.tight_layout()
            points = plt.ginput(7, timeout=0, mouse_pop=2)
            plt.close()
            
            creation_time = os.path.getmtime(file_path)
            formatted_datetime = datetime.datetime.fromtimestamp(creation_time).strftime("%m/%d/%Y %H:%M")

            new_rows = []
            for k, stake in enumerate(STAKES):
                top = points[2 * k]
                bottom = points[2 * k + 1]
                pixel_length = math.dist(top, bottom)
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