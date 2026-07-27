'''
Original author: Catherine Breen (July 1, 2024)
Updated by: Kent Pawson (2026) Adapted for multi-pole keypoint configuration and custom dataset pipelines.

Training script for users to fine tune model from Breen et. al 2024
Please cite: 

Breen, C. M., Currier, W. R., Vuyovich, C., Miao, Z., & Prugh, L. R. (2024). 
Snow Depth Extraction From Time‐Lapse Imagery Using a Keypoint Deep Learning Model. 
Water Resources Research, 60(7), e2023WR036682. https://doi.org/10.1029/2023WR036682

python src/train.py

tensorboard --logdir=runs

'''

# Import startup libraries
import os

# Import all libraries
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib
import utils
from model import snowPoleResNet50
from tqdm import tqdm
import numpy as np
from dataset import train_data, train_loader, validation_data, validation_loader
# training viz 
from torch.utils.tensorboard import SummaryWriter  # For PyTorch
import copy
from pathlib import Path
import pandas as pd

from arg_parser import ArgumentParser

# Argument parser
args = ArgumentParser("Train a model on a set of images", "train")

matplotlib.style.use('ggplot')
# start_time = time.time() 

#TODO: figure out if there's a better place to write summary
writer = SummaryWriter(f'runs/CHRL_2026')

## create output path
if not os.path.exists(f"{args.models_output}"):
    os.makedirs(f"{args.models_output}", exist_ok=True)

labels_path = Path(args.path) / "labels.csv"
df_labels = pd.read_csv(labels_path)

#Determine how many active poles we have, based on how many _x1 columns are present in labels.csv
pole_x1_cols = [col for col in df_labels.columns if col.endswith("_x1")]
num_poles = len(pole_x1_cols)

# model
num_keypoints = 4 * num_poles
model = snowPoleResNet50(pretrained=True, requires_grad=True, num_keypoints=num_keypoints).to(args.device)

checkpoint = torch.load(args.model_path, map_location=torch.device(args.device), weights_only=False)

state_dict = checkpoint["model_state_dict"]

#Remove mismatched output layers (due to different number of snow poles)
state_dict.pop("l0.weight", None)
state_dict.pop("l0.bias", None)

model.load_state_dict(state_dict, strict=False)

print("fine-tuned model loaded...")

# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5, min_lr=1e-6)
criterion = nn.SmoothL1Loss()

# training function
def fit(model, dataloader, data):

    # print("Training")
    # model.to(args.device)  ##

    #.embed()
    print('Training')
    model.to(args.device) ##

    model.train()
    train_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1

        image, keypoints = data["image"].to(args.device), data["keypoints"].to(
            args.device
        )

        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, keypoints)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss/counter
    return train_loss


# validation function
def validate(model, dataloader, data, epoch):
    print("Validating")
    model.to(args.device)

    model.eval()
    valid_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1

            image, keypoints = data["image"].to(args.device), data["keypoints"].to(
                args.device
            )

            outputs = model(image)
            loss = criterion(outputs, keypoints) ## cross entropy loss between input and output
            valid_running_loss += loss.item()
            # plot the predicted validation keypoints after every...
            # ... predefined number of epochs
            if not os.path.exists(args.models_output):
                os.makedirs(args.models_output, exist_ok=True)
            if (epoch + 1) % 10 == 0:
                utils.valid_keypoints_plot(args, image, outputs, keypoints, epoch)
        
    valid_loss = valid_running_loss/counter
    return valid_loss

print("Training Starting...")
train_loss = []
val_loss = []
## early stopping ##
#######################
best_loss_val = np.inf
best_loss_val_epoch = 0 
best_model_weights = copy.deepcopy(model.state_dict())
#######################
for epoch in range(args.epochs):

    print(f"Epoch {epoch+1} of {args.epochs}")
    train_epoch_loss = fit(model, train_loader, train_data)
    val_epoch_loss = validate(model, validation_loader, validation_data, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {val_epoch_loss:.4f}')
    ####### saving model every 50 epochs
    if (epoch % 50) == 0:
        torch.save(
            {
                "epoch": args.epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": criterion,
            },
            f"{args.models_output}/model_epoch{epoch}.pth",
        )

    writer.add_scalar('Loss/train',train_epoch_loss, epoch)
    writer.add_scalar('Loss/validation',val_epoch_loss, epoch)
    writer.flush()

    ####### early stopping #########
    if val_epoch_loss < best_loss_val:
                best_loss_val = val_epoch_loss
                best_loss_val_epoch = epoch
                best_model_weights = copy.deepcopy(model.state_dict())
    elif epoch > best_loss_val_epoch + 25:
            ### save model at lowest val error, rather than 10 epochs later 
            model.load_state_dict(best_model_weights)
            break
    
    scheduler.step(val_epoch_loss)

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# Add white background and light grey grid
plt.gca().set_facecolor('white')
plt.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)
plt.gca().set_axisbelow(True)  # Put grid lines behind the plot lines
##

plt.savefig(f"{args.models_output}/loss.png")
plt.close()  # changed from plt.show()
torch.save(
    {
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": criterion,
    },
    f"{args.models_output}/model.pth",
)  ### the last model
writer.close()
print("DONE TRAINING")

# print("My program took", time.time() - start_time, "to run")
