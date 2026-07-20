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
import math
import pandas as pd
import numpy as np
import os
import datetime
from pathlib import Path
from utils import enable_scroll_zoom_and_pan
from arg_parser import ArgumentParser
from config import cameras, labeling

def get_subset_to_label(files):
    target = labeling.get("target_label_count", 25)
    total = len(files)

    #If total files are less than target amount, then just use all of them
    if total <= target:
        return files
    
    #Generate evenly spaced indices from start to very last file
    indices = np.linspace(0, total - 1, target, dtype=int)

    return [files[i] for i in indices]

def main():
    # Argument parser for command-line arguments:
    args = ArgumentParser("Manually label images for training", "label")

    metadata_path = Path(args.path) / "pole_metadata.csv"
    meta_df = pd.read_csv(metadata_path)

    ## customized data
    #pole_length = np.float64(args.pole_length)
    # subset_to_label = np.int16(args.subset_to_label)

    ## load labels.csv
    labels_path = Path(args.path) / "labels.csv"
    CORE_COLUMNS = ["filename", "camera_id", "datetime"]

    #Check if labels file exists.
    if labels_path.exists():
        #If file exists check if required columns exist. If not, create new labels file with required columns
        try:
            df_existing = pd.read_csv(labels_path, skip_blank_lines=True)
            if not all(col in df_existing.columns for col in CORE_COLUMNS):
                raise ValueError("Missing required columns")
            
        except Exception:
            print("labels.csv is corrupted or does not exist, creating...")
            df_existing = pd.DataFrame(columns=CORE_COLUMNS)
            df_existing.to_csv(labels_path, index=False)

    else:
        #If file does not exist, create new one
        print("labels.csv is corrupted or does not exist, creating...")
        df_existing = pd.DataFrame(columns=CORE_COLUMNS)
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
        camera_cfg = cameras.get(camera_id)

        if not camera_cfg:
            continue #skip if camera id is not in configuration

        is_enabled = camera_cfg.get("enabled", True)
        active_poles = camera_cfg.get("active_poles", [])

        if not is_enabled or not active_poles:
            continue # skip if camera is not enabled or if it has no active poles

        try:
            cam_meta = meta_df[meta_df["camera_id"] == camera_id].iloc[0]
        except IndexError:
            print(f"Warning: No calibration data found in pole_metadata.csv for camera: {camera_id}. Skipping directory.")
            continue
        
        files = sorted(cam_dir.glob("*.JPG"))
        subset_files = get_subset_to_label(files)
        total_to_label = len(subset_files)

        for current_idx, file_path in enumerate(subset_files):
            
            #Avoid labeling images that have already been labeled
            if(file_path.name, camera_id) in already_labeled:
                print(f'{file_path.name} already labeled, skipping to next subset')
                continue
            
            img = cv2.imread(str(file_path))

            #If camera site is marked as upside-down, flip image
            if camera_cfg.get("upside_down", False):
                img = cv2.rotate(img, cv2.ROTATE_180)

            figure = plt.figure(figsize=(16, 8), num=file_path.name)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax = plt.gca()
            enable_scroll_zoom_and_pan(ax)

            expected_clicks = 2 * len(active_poles) + 1
            current_label_num = current_idx + 1

            plt.title(
                f"$\\bf{{{camera_id}}}$ | ACTIVE POLES: {active_poles} | PROGRESS: {current_label_num}/{total_to_label}\n"
                f"1. Click top & bottom of each active pole (left to right)\n"
                f"[$\\bf{{Backspace:}}$ Undo last click | $\\bf{{Scroll:}}$ Zoom | $\\bf{{Right-Click + Drag:}}$ Pan | $\\bf{{Enter:}}$ Skip image]",
                fontsize=12, 
                color="#1a1a1a",
                pad=10,
                loc="left", 
                ma="left"
            )
            
            plt.tight_layout()
            points = plt.ginput(expected_clicks, timeout=0, mouse_pop=2)
            plt.close()

            if len(points) < expected_clicks:
                print(f"Skipping {file_path.name} (Not enough points collected / User skipped)")
                continue

            creation_time = os.path.getmtime(file_path)
            formatted_datetime = datetime.datetime.fromtimestamp(creation_time).strftime("%m/%d/%Y %H:%M")

            new_row = {
                "filename": file_path.name,
                "camera_id": camera_id,
                "datetime": formatted_datetime,
            }
    
            for j, poleId in enumerate(active_poles):
                top = points[2 * j]
                bottom = points[2 * j + 1]
                pixel_length = math.dist(top, bottom)

                #Attempt to calculate snow depth
                try:
                    #snow depth calculated by subtracting the known pole length by our pixel length (where we clicked in the image) * our conversion (previously calculated)
                    #In snow free images, snow depth should be ~0, which can be used as ground truth.
                    #In snowy images, our bottom click position should be at the snow interface, and the snow depth value should be higher (depending on amount of snow).
                    pixel_length_cm = float(cam_meta[f"s{poleId}_pole_length_cm"])
                    pixel_cm_conversion = float(cam_meta[f"s{poleId}_pixel_cm_conversion"])
                    snow_depth = pixel_length_cm - (pixel_length * pixel_cm_conversion)
                
                except (KeyError, ValueError, TypeError) as e:
                    print(f"  [Warning] Missing/invalid calibration metadata for {camera_id} Pole {poleId}. Skipping snow depth calculation.")
                    snow_depth = None

                new_row[f"s{poleId}_x1"] = top[0]
                new_row[f"s{poleId}_y1"] = top[1]
                new_row[f"s{poleId}_x2"] = bottom[0]
                new_row[f"s{poleId}_y2"] = bottom[1]
                new_row[f"s{poleId}_pixel_length"] = pixel_length
                new_row[f"s{poleId}_snow_depth"] = snow_depth

            #Create new dataframe containing dynamic columns for the new row and concat with existing dataframe
            df_new_row = pd.DataFrame([new_row])
            df_existing = pd.concat([df_existing, df_new_row], ignore_index=True)
            #Save dataframe to labels file
            df_existing.to_csv(labels_path, index=False)
            print(f"Successfully saved wide row metrics for: {file_path.name}")

if __name__ == "__main__":
    main()