'''
Original author: Catherine Breen (July 1, 2024)
Updated by: Kent Pawson (2026) Adapted for multi-pole keypoint configuration and custom dataset pipelines.

load model and run on data points 
export the csv of the data points and just use the bottom

example command line to run:

(make sure config file is set to the right model!)
python src/evaluate.py

'''
import os
import numpy as np
import utils
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import distance
import os
from arg_parser import ArgumentParser
from config import cameras, global_max_poles, paths
print("Initializing environment and loading ML libraries (this may take a few seconds)...", flush=True)
from model import snowPoleResNet50
from dataset import SnowPoleDataset
import torch

def load_model(args):
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


def predict(args, model, data): 
    evaluation_data = {
        "camera": [],
        "filename": []
    }

    metadata =  pd.read_csv(f"{args.path}/pole_metadata.csv")
    labels =  pd.read_csv(f"{args.path}/labels.csv")

    with torch.no_grad():
        for i, data in tqdm(enumerate(data)): 
            image, keypoints = data['image'].to(args.device), data['keypoints'].to(args.device)
            filename = data['filename']
            camera_id = filename.split('_')[0]
            camera_cfg = cameras.get(camera_id)

            if not camera_cfg:
                continue

            active_poles = camera_cfg.get("active_poles", [])
            is_enabled = camera_cfg.get("enabled", True)

            if not is_enabled or not active_poles:
                continue

            current_row_idx = len(evaluation_data["filename"])
            
            evaluation_data["camera"].append(camera_id)
            evaluation_data['filename'].append(filename)

            ## add an empty dimension for sample size
            image = image.unsqueeze(0)
            outputs = model(image)
            outputs = outputs.detach().cpu().numpy()
            pred_keypoint = np.array(outputs[0], dtype='float32')

            # flatten the keypoints
            keypoints = keypoints.detach().cpu().numpy().reshape(-1,2)

            utils.eval_keypoints_plot(args, filename, image, outputs, orig_keypoints=keypoints) ## visualize points
            
            for j, poleId in enumerate(active_poles):
                
                x1_true = keypoints[2*j, 0]
                y1_true = keypoints[2*j, 1]
                x2_true = keypoints[2*j + 1, 0]
                y2_true = keypoints[2*j + 1, 1]

                x1_pred = pred_keypoint[4*j + 0]
                y1_pred = pred_keypoint[4*j + 1]
                x2_pred = pred_keypoint[4*j + 2]
                y2_pred = pred_keypoint[4*j + 3]
                
                ## 224 canvas pixel distances
                total_length_pixel_224 = distance.euclidean([x1_pred, y1_pred], [x2_pred, y2_pred])
                total_length_pixel_actual_224 = distance.euclidean([x1_true, y1_true], [x2_true, y2_true])

                try:
                    camera_row = metadata[metadata['camera_id'] == camera_id].iloc[0]
                    full_length_pole_cm = float(camera_row[f"s{poleId}_pole_length_cm"])
                    pixel_cm_conversion = float(camera_row[f"s{poleId}_pixel_cm_conversion"])

                    img_row = labels[labels['filename'] == filename]
                    manual_pixel_length = img_row[f's{poleId}_pixel_length'].values[0]
                    manual_snowdepth = img_row[f's{poleId}_snow_depth'].values[0] #full_length_pole_cm - (pixel_cm_conversion * manual_pixel_length)
                    
                    #We scale 224 predicted pixel length up to match full resolution (which our px to cm conversions are based on)
                    scale_ratio = manual_pixel_length / total_length_pixel_actual_224
                    total_length_pixel_full_res = total_length_pixel_224 * scale_ratio

                    automated_sd = full_length_pole_cm - (pixel_cm_conversion * total_length_pixel_full_res)
                    difference = manual_snowdepth - automated_sd

                    ## error
                    top_pixel_error = distance.euclidean([x1_true, y1_true], [x1_pred, y1_pred])
                    bottom_pixel_error = distance.euclidean([x2_true, y2_true], [x2_pred, y2_pred])

                    # MAPE
                    mape_error = utils.MAPE(total_length_pixel_actual_224, total_length_pixel_224)

                    #Only calculate MAPE if there is actually enough snow to make a meaningful percentage
                    if manual_snowdepth > 5.0:
                        mape_error_sd = utils.MAPE(manual_snowdepth, automated_sd)
                    else:
                        mape_error_sd = None

                except Exception as e:
                    print(f"Error computing error metrics for {filename}, Pole {poleId}: {e}")
                    #Fallback, set all values to 0
                    full_length_pole_cm, pixel_cm_conversion = 0, 0
                    manual_pixel_length, manual_snowdepth, difference = 0, 0, 0
                    top_pixel_error, bottom_pixel_error = 0, 0
                    mape_error, mape_error_sd = 0, 0
                    automated_sd = 0

                #Append flat dynamic row into evaluation data
                for key, val in [
                    (f"s{poleId}_x1_true", x1_true),
                    (f"s{poleId}_y1_true", y1_true),
                    (f"s{poleId}_x2_true", x2_true),
                    (f"s{poleId}_y2_true", y2_true),

                    (f"s{poleId}_x1_pred", x1_pred),
                    (f"s{poleId}_y1_pred", y1_pred),
                    (f"s{poleId}_x2_pred", x2_pred),
                    (f"s{poleId}_y2_pred", y2_pred),

                    (f"s{poleId}_total_length_pixel", total_length_pixel_224),
                    (f"s{poleId}_full_length_pole_cm", full_length_pole_cm),
                    (f"s{poleId}_pixel_cm_conversion", pixel_cm_conversion),
                    
                    (f"s{poleId}_automated_sd", automated_sd),

                    (f"s{poleId}_manual_pixel_length", manual_pixel_length),
                    (f"s{poleId}_manual_snowdepth", manual_snowdepth),
                    (f"s{poleId}_difference_cm", difference),
                    (f"s{poleId}_top_pixel_error", top_pixel_error),
                    (f"s{poleId}_bottom_pixel_error", bottom_pixel_error),
                    (f"s{poleId}_mape", mape_error),
                    (f"s{poleId}_mape_sd", mape_error_sd),
                ]:
                    if key not in evaluation_data:
                        evaluation_data[key] = [None] * current_row_idx
                    evaluation_data[key].append(val)

            target_length = current_row_idx + 1
            for key in evaluation_data.keys():
                if len(evaluation_data[key]) < target_length:
                    evaluation_data[key].append(None)
    
    results = pd.DataFrame(evaluation_data)
    results.to_csv(f"{args.models_output}/eval/indiv_img_eval_results.csv")

    metric_substrings = {
        "Top Pixel Error": "_top_pixel_error",
        "Bottom Pixel Error": "_bottom_pixel_error",
        "Mean Average Percent Error (MAPE)": "_mape",
        "Difference in cm": "_difference_cm",
        "Difference in MAPE": "_mape_sd",
    }

    stats_data = {
        "Site": [],
        "Pole": [],
        "Metric": [],
        "Mean": [],
        "Standard_Deviation": [],
    }

    for camera_id, group_df in results.groupby('camera'):
        # print("#### Overall average\n")
        print(f'Processing metrics for {camera_id}')

        for metric_label, substring in metric_substrings.items():
            #Find all columns in group_df containing substring
            metric_cols = [col for col in group_df.columns if col.endswith(substring)]

            for col in metric_cols:
                pole_prefix = col.split('_')[0].upper()

                all_values = group_df[col].to_numpy()
                #Remove any None or Nan
                all_values = all_values[pd.notnull(all_values)]
                all_values = all_values[~np.isnan(all_values.astype(float))]

                if len(all_values) > 0:
                    mean_val = round(np.mean(all_values), 3)
                    std_val = round(np.std(all_values), 3)

                    stats_data['Site'].append(camera_id)
                    stats_data['Pole'].append(pole_prefix)
                    stats_data['Metric'].append(metric_label)
                    stats_data['Mean'].append(mean_val)
                    stats_data['Standard_Deviation'].append(std_val)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(stats_data)
    df.to_csv(f"{args.models_output}/eval/overall_statistics.csv", index=False)

    return results

def main():
    # Argument parser
    args = ArgumentParser("Evaluate model on the train/val images")

    if not os.path.exists(f"{args.models_output}/eval"):
        os.makedirs(f"{args.models_output}/eval", exist_ok=True)

    model = load_model(args)
    
    all_images = list(Path(paths.get('data_directory')).rglob("*.JPG"))
    parents_dict = {i.name: str(i) for i in all_images}
    validation_samples = pd.read_csv(f"{args.models_output}/training/validation_samples.csv")
    
    validation_data = SnowPoleDataset(
        validation_samples, 
        parents_dict, 
        aug=False
    )

    print('results on valid data\n')
    outputs = predict(args, model, validation_data)

    print("Evaluation complete, results saved.")

if __name__ == '__main__':
    main()



