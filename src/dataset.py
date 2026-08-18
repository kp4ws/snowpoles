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
import torch.nn.functional as F
import cv2
import numpy as np
import utils
from torch.utils.data import Dataset
import torch
import albumentations as A ### better for keypoint augmentations, pip install albumentations
from config import cameras, paths, training, global_max_poles

class SnowPoleDataset(Dataset):
    def __init__(self, samples, parents_dict, aug=False): # split='train'):
        self.data = samples
        self.parents = parents_dict
        self.resize = 224

        if aug == False: 
            self.transform = A.Compose([
                A.Resize(self.resize, self.resize),
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
                A.Resize(self.resize, self.resize),
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
        image = cv2.imread(self.parents[self.data.iloc[index]["filename"]])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if camera_cfg.get("upside_down", False):
            image = cv2.rotate(image, cv2.ROTATE_180)
        
        orig_h, orig_w, channel = image.shape

        #Scale pixel values to [0, 1] and resize to target resolution
        image = image / 255.0
        # resize the image into `resize` defined above
        image = cv2.resize(image, (self.resize, self.resize))

        #Apply HSV snow filter (if turned on in config)
        if training.get('filter'): 
            image = utils.apply_filter(image)
            if index % 100 == 0: 
                cv2.imwrite(f"{paths.get('models_output')}/filtered_{filename}", image)

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

        #Scale coordinates to match new canvas dimensions
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

        #padding logic and flatten back to vector of 12 values for output layer
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float).view(-1)
        max_tensor_size = global_max_poles * 4
        padding_length = max_tensor_size - keypoints_tensor.shape[0]
        if padding_length > 0:
            keypoints_tensor = F.pad(keypoints_tensor, (0, padding_length), value = -999)

        return {
            'image': torch.tensor(image, dtype=torch.float),
            'keypoints': keypoints_tensor,
            'filename': filename
        }