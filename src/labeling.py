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

    # dir = glob.glob(f"{args.path}/**/*")  # /*") ## path to data directory
    dir = list(
        Path(args.path).rglob("*.JPG")
    )  # Recursively lists all files and directories
    dir = sorted(dir)

    ## labeling data
    camera_ids = []
    stake_ids = []
    filenames = []
    creation_times = []
    topX, topY, bottomX, bottomY = [], [], [], []
    pixel_lengths = []
    snow_depths = []

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
    #snowdepths = df_existing["snow_depth"].tolist()

    # Reset stake_ids for the main labeling loop
    stake_ids = []

    ### loop to label every nth photo!
    i = 0
    prev_cameraID = ""
    for j, file in tqdm.tqdm(enumerate(dir)):
        cameraID = Path(file).parent.name
        # whether to start counter over
        #i = i if len(cameraids) == 1 or cameraID == cameraids[-2] else 0
        if j == 0 or cameraID != Path(dir[j-1]).parent.name:
            i = 0

        if Path(file).name in filenames:
            print(" ", Path(file).name, "has been labeled before, using stored data.")

        if i % subset_to_label == 0 and (not Path(file).name in filenames):
            camera_ids.append(cameraID)
            print(" ", Path(file).name)
            img = cv2.imread(str(file))

            for k, stake in enumerate(STAKES):
                height, width, channel = img.shape
                ## assumes the cameras are stored in folder with their camera name
                figure = plt.figure(figsize=(20, 10), num=Path(file).name)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                ax = plt.gca()
                enable_scroll_zoom_and_pan(ax)

                plt.title("label top and then bottom of full pole \n Click ANYWHERE to confirm | BACKSPACE to undo | RIGHT-CLICK drag | SCROLL zoom.", fontweight="bold")
                points = plt.ginput(3, timeout=0, mouse_pop=2)
                top, bottom = points[0], points[1]
                topX.append(top[0]), topY.append(top[1])
                bottomX.append(bottom[0]), bottomY.append(bottom[1])
                plt.close()

                pixel_length = math.dist(top, bottom)
                pixel_lengths.append(pixel_length)

                stake_ids.append(stake)

            ## save data to labels.csv
            nextline = f"\n{Path(file).name},{os.path.getctime(file)},{top[0]},{top[1]},{bottom[0]},{bottom[1]},{pixel_length}"
            with open(f"{args.path}/labels.csv", "a") as labels2_csv:
                labels2_csv.write(nextline)

            filenames.append(Path(file).name)
            creationTime = os.path.getmtime(file)
            dt_c = datetime.datetime.fromtimestamp(creationTime)
            formatted_datetime = dt_c.strftime("%m/%d/%Y %H:%M")
            creation_times.append(formatted_datetime)

            ## snowdepth ##
            snow_depth = pole_length_cm_lookup[cameraID] - (pixel_length * conversion_lookup[cameraID])
            snow_depths.append(snow_depth)

        i += 1

    ## simplified table for snow depth conversion later on
    df = pd.DataFrame(
        {
            "camera_id":camera_ids,
            "filename": filenames,
            "datetime": creation_times,
            "x1": topX,
            "y1": topY,
            "x2": bottomX,
            "y2": bottomY,
            "pixel_lengths": pixel_lengths,
            "snow_depths":snow_depths,
        }
    )

    df.to_csv(f"{args.path}/labels.csv", index=False)

if __name__ == "__main__":
    main()