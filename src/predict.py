'''
Original author: Catherine Breen (July 1, 2024)
Updated by: Kent Pawson (2026) Adapted for multi-pole keypoint configuration and custom dataset pipelines.

Training script for users to fine tune model from Breen et. al 2024
Please cite: 

Breen, C. M., Currier, W. R., Vuyovich, C., Miao, Z., & Prugh, L. R. (2024). 
Snow Depth Extraction From Time‐Lapse Imagery Using a Keypoint Deep Learning Model. 
Water Resources Research, 60(7), e2023WR036682. https://doi.org/10.1029/2023WR036682

python src/predict.py

'''

# Import startup libraries
import os
from pathlib import Path

# for datetime
import datetime

# for predict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
import torch
from tqdm import tqdm
# Import all libraries
from model import snowPoleResNet50
import torch
from config import cameras, global_max_poles

from arg_parser import ArgumentParser

def vis_predicted_keypoints(file, image, keypoints, color=(0, 255, 0), diameter=15, active_poles=3, args=None):
    #file = file.split(".")[0]
    file = Path(file).stem  
    output_keypoint = keypoints.reshape(-1, 2)
    plt.imshow(image)

    num_active_points = active_poles * 2
    for p in range(num_active_points):
        if output_keypoint[p, 0] == -999.0 or output_keypoint[p, 1] == -999.0:
            continue

        plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.')

    plt.savefig(f"{args.models_output}/predictions/pred_{file}.png")
    plt.close()

def load_model(args):
    labels_path = Path(args.path) / "labels.csv"
    df_labels = pd.read_csv(labels_path)

    num_keypoints = 4 * global_max_poles
    model = snowPoleResNet50(pretrained=False, requires_grad=False, num_keypoints=num_keypoints).to(args.device)
    # load the model checkpoint
    #torch.serialization.add_safe_globals([torch.nn.modules.loss.SmoothL1Loss])
    model_path = args.model_path
    checkpoint = torch.load(model_path, map_location=torch.device(args.device), weights_only=False)

    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def predict(model, args, device):  
    if not os.path.exists(f"{args.models_output}/predictions"):
        os.makedirs(f"{args.models_output}/predictions", exist_ok=True)

    predictions_data = {
        "camera_id": [],
        "filename": [],
        "datetime": [],
    }

    ## folder or directory
    sites_dir = Path(args.path) / "sites"
    snowpolefiles = list(sites_dir.rglob("*.JPG"))
    metadata = pd.read_csv(f"{args.path}/pole_metadata.csv")

    with torch.no_grad():
        for i, file in tqdm(enumerate(snowpolefiles)): 
            file_path = Path(file)
            filename = file_path.name
            camera_id = file_path.stem.split('_')[0]
            camera_cfg = cameras.get(camera_id)

            if not camera_cfg:
                continue

            active_poles = camera_cfg.get("active_poles", [])
            is_enabled = camera_cfg.get("enabled", True)

            if not is_enabled or not active_poles:
                continue

            image = cv2.imread(str(file))
            if camera_cfg.get("upside_down", False):
                image = cv2.rotate(image, cv2.ROTATE_180)

            creationTime = os.path.getmtime(file)
            dt_c = datetime.datetime.fromtimestamp(creationTime)
            formatted_datetime = dt_c.strftime("%m/%d/%Y %H:%M")
            
            current_row_idx = len(predictions_data["camera_id"])

            predictions_data["camera_id"].append(camera_id)
            predictions_data["filename"].append(filename)
            predictions_data['datetime'].append(formatted_datetime)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, *_ = image.shape
            image = cv2.resize(image, (224,224))
            image = image / 255.0   

            # again reshape to add grayscale channel format
            
            ## add an empty dimension for sample size
            image = np.transpose(image, (2, 0, 1)) ## 
            image = torch.tensor(image, dtype=torch.float).unsqueeze(0).to(device)

            #######
            outputs = model(image)
            outputs = outputs.cpu().numpy() 
            pred_keypoint = np.array(outputs[0], dtype='float32')

            image = image.squeeze()
            image = image.cpu()
            image = np.transpose(image, (1, 2, 0))
            image = np.asarray(image, dtype='float32')

            ## resize back up to original size and project predicted points onto original size
            image = cv2.resize(image, (w, h))
            
            for j, poleId in enumerate(active_poles):
                base = 4*j

                x1_pred = pred_keypoint[base + 0] * (w / 224)
                y1_pred = pred_keypoint[base + 1] * (h / 224)
                x2_pred = pred_keypoint[base + 2] * (w / 224)
                y2_pred = pred_keypoint[base + 3] * (h / 224)

                pred_keypoint[base + 0] = x1_pred
                pred_keypoint[base + 1] = y1_pred
                pred_keypoint[base + 2] = x2_pred
                pred_keypoint[base + 3] = y2_pred

                total_length_pixel = distance.euclidean([x1_pred, y1_pred], [x2_pred, y2_pred])
                
                try:
                    ## snow depth conversion ## 
                    camera_row = metadata[metadata['camera_id'] == camera_id].iloc[0]
                    full_length_pole_cm = float(camera_row[f"s{poleId}_pole_length_cm"])
                    pixel_cm_conversion = float(camera_row[f"s{poleId}_pixel_cm_conversion"])
                    snow_depth = full_length_pole_cm - (pixel_cm_conversion * total_length_pixel)
                except Exception:
                    # Fallback default values if metadata lookup or lookup row fails
                    full_length_pole_cm, pixel_cm_conversion, snow_depth = 0, 0, 0
                
                for key, val in [
                    (f"s{poleId}_x1_pred", x1_pred),
                    (f"s{poleId}_y1_pred", y1_pred),
                    (f"s{poleId}_x2_pred", x2_pred),
                    (f"s{poleId}_y2_pred", y2_pred),
                    (f"s{poleId}_total_length_pixel", total_length_pixel),
                    (f"s{poleId}_snow_depth", snow_depth),
                ]:
                    if key not in predictions_data:
                        predictions_data[key] = [None] * current_row_idx
                    predictions_data[key].append(val)

            #Ensures all dictionary arrays are the same size
            target_length = current_row_idx + 1
            for key in predictions_data.keys():
                if len(predictions_data[key]) < target_length:
                    predictions_data[key].append(None)

            if i % 100 == 0: 
                vis_predicted_keypoints(filename, image, pred_keypoint, active_poles, args=args) 

            if camera_cfg.get("upside_down", False):
                x1_pred = w - x1_pred
                y1_pred = h - y1_pred
                x2_pred = w - x2_pred
                y2_pred = h - y2_pred
            
    results = pd.DataFrame(predictions_data)
    results.to_csv(f"{args.models_output}/predictions/results.csv")

    return results

def main():
    args = ArgumentParser("Use a model to predict snow depth")
    model = load_model(args)
    device = 'cpu'
    predict(model, args, device)  

if __name__ == '__main__':
    main()