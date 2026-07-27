'''
Original author: Catherine Breen (July 1, 2024)
Updated by: Kent Pawson (2026) Adapted for multi-pole keypoint configuration and custom dataset pipelines.

Training script for users to fine tune model from Breen et. al 2024
Please cite: 

Breen, C. M., Currier, W. R., Vuyovich, C., Miao, Z., & Prugh, L. R. (2024). 
Snow Depth Extraction From Time‐Lapse Imagery Using a Keypoint Deep Learning Model. 
Water Resources Research, 60(7), e2023WR036682. https://doi.org/10.1029/2023WR036682


'''
import torch
import cv2
import pandas as pd
import numpy as np
import tomli as tomllib
import utils
from torch.utils.data import Dataset, DataLoader
import torch
import albumentations as A ### better for keypoint augmentations, pip install albumentations
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from config import cameras

# Load config from config.toml
with open("config.toml", "rb") as configfile:
    config = tomllib.load(configfile)

# Load config from config.toml
with open("config.toml", "rb") as configfile:
    config = tomllib.load(configfile)


def apply_filter(image):
    # width, height, __ = image.shape
    # for y in range(height):
    #     for x in range(width):
    #         pixel = list(colorsys.rgb_to_hsv(*image[x, y]))
    #         if (pixel[0] < 0.833):
    #             image[x, y] = (0, 0, 0)
    #             continue
    #         pixel[1] = 1
    #         pixel[2] = 255
    #         rgb = colorsys.hsv_to_rgb(*pixel)
    #         image[x, y] = (round(rgb[0]), round(rgb[1]), round(rgb[2]))
    image_rgb = image[:, :, ::-1]
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = image_hsv[:, :, 0] < 149
    image_rgb[mask] = [0,0,0]
    image_hsv[~mask, 1] = 255
    image_hsv[~mask, 2] = 255
    valid_pixels = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    image_rgb[~mask] = valid_pixels[~mask]
    #print("filtered applied!")
    return image_rgb[:, :, ::-1]

    

# Define a function to sample every third photo
## Only used for experiments 
def sample_every_x(group, x):
    indices = np.arange(len(group[1]))
    every_x = len(group[1])//x
    selected_indices = indices[2::every_x]  
    return group[1].iloc[selected_indices]

def train_test_split(csv_path, image_path):

    df_data = pd.read_csv(csv_path)
    print(f'all rows in df_data {len(df_data.index)}')

    ## check to make sure we only use images that exist
    all_images = list(Path(image_path).rglob("*.JPG"))

    global parents
    parents = {}
    for i in all_images:
        parents[i.name] = str(i)
    existing_filenames = [img.name for img in all_images]
    
    #Filter df_data to ensure we only have existing filenames
    df_existing = df_data[df_data["filename"].isin(existing_filenames).reset_index(drop=True)]

    #Perform 80/20 split on training and validation data
    training_samples = df_existing.sample(frac=0.8, random_state=100) # same shuffle everytime
    validation_samples = df_existing[~df_existing.index.isin(training_samples.index)]

    #Reset indices
    training_samples = training_samples.reset_index(drop=True)
    validation_samples = validation_samples.reset_index(drop=True)
    
    # save labels to output folder
    if not os.path.exists(f"{config['paths']['models_output']}"):
        os.makedirs(f"{config['paths']['models_output']}", exist_ok=True)
    training_samples.to_csv(f"{config['paths']['models_output']}/training_samples.csv")
    validation_samples.to_csv(f"{config['paths']['models_output']}/validation_samples.csv")

    print(f'# of examples we will now train on {len(training_samples)}, val on {len(validation_samples)}')
    return training_samples, validation_samples

class snowPoleDataset(Dataset):

    def __init__(self, samples, path, aug): # split='train'):
        self.data = samples
        self.path = path
        self.resize = 224

        if aug == False: 
            self.transform = A.Compose([
                A.Resize(224, 224),
                ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else: 
            self.transform = A.Compose([
                A.ToFloat(max_value=1.0),
                # A.CropAndPad(px=50, p =1.0), ## final model is 50 pixels
                # A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=5, p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                    A.ToGray(p=0.5)], p = 0.5),
                A.Resize(224, 224),
                ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self):
        return len(self.data)

    def __filename__(self, index):
        return self.data.iloc[index]['filename']
    
    def __getitem__(self, index):
        #Grab row for given index
        row = self.data.iloc[index]
        filename = row['filename']
        cameraID = row['camera_id']
        camera_cfg = cameras.get(cameraID)
        active_poles = camera_cfg.get("active_poles", [])

        #Read and prepare image
        image = cv2.imread(parents[self.data.iloc[index]["filename"]])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if camera_cfg.get("upside_down", False):
            image = cv2.rotate(image, cv2.ROTATE_180)
        
        orig_h, orig_w, channel = image.shape

        #Scale pixel values to [0, 1] and resize to 224x224
        image = image / 255.0
        # resize the image into `resize` defined above
        image = cv2.resize(image, (self.resize, self.resize))

        #Apply HSV snow filter (if turned on in config)
        if config['training']['filter']: 
            image = apply_filter(image)
            if index % 100 == 0: 
                cv2.imwrite(f"{config['paths']['models_output']}/filtered_{filename}", image)

        #The columns that correspond to the keypoints in our dataframe
        keypoint_columns = []
        for poleId in active_poles:
            keypoint_columns.extend([
                f's{poleId}_x1', f's{poleId}_y1', 
                f's{poleId}_x2', f's{poleId}_y2'
            ])

        # #Clip negative noise values to 0 and convert to float32 type
        keypoints = row[keypoint_columns].clip(lower=0).values.astype('float32')

        # #Reshape from a flat 12 element array to (6, 2) matrix for Albumentations
        keypoints = keypoints.reshape(-1, 2)

        #Scale coordinates to match new 224x224 canvas dimensions
        keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h]

        #Pass image and 6 keypoints to albumentations
        transformed = self.transform(image=image, keypoints=keypoints)
        img_transformed = transformed['image']
        keypoints = transformed['keypoints']

        # viz training data
        #utils.vis_keypoints(transformed['image'], transformed['keypoints'])
        image = np.transpose(img_transformed, (2, 0, 1))

        expected_points = 2 * len(active_poles)
        if len(keypoints) != expected_points:
            utils.vis_keypoints(transformed['image'], transformed['keypoints'])

        return {
            'image': torch.tensor(image, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float).view(-1), #Flatten back to vector of 12 values for output layer
            'filename': filename
        }

# get the training and validation data samples
training_samples, validation_samples = train_test_split(
    f"{config['paths']['labels']}", config['paths']['input_images']
)

#{config['paths']['input_images']}/

# initialize the dataset - `snowPoleDataset()`
train_data = snowPoleDataset(
    training_samples,
    f"{config['paths']['input_images']}",
    aug=config['training']['aug'],
)  ## we want all folders

validation_data = snowPoleDataset(
    validation_samples, 
    f"{config['paths']['input_images']}", 
    aug=False
)  # we always want the transform to be the normal transform

# # prepare data loaders
train_loader = DataLoader(
    train_data, 
    batch_size=config['training']['batch_size'], 
    shuffle=True, 
    num_workers=0
)
validation_loader = DataLoader(
    validation_data,
    batch_size=config['training']['batch_size'],
    shuffle=False,
    num_workers=0,
)

print(f"Training sample instances: {len(train_data)}")
print(f"Validation sample instances: {len(validation_data)}")

if config["training"]["show_dataset_plot"]:
    utils.dataset_keypoints_plot(train_data)
    utils.dataset_keypoints_plot(validation_data)