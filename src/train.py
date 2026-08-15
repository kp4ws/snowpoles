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
from dataset import SnowPoleDataset
# training viz 
from torch.utils.tensorboard import SummaryWriter  # For PyTorch
from torch.utils.data import DataLoader
import copy
from pathlib import Path
import pandas as pd
from config import global_max_poles, paths, training
from arg_parser import ArgumentParser

def load_model(args):
    num_keypoints = 4 * global_max_poles
    model = snowPoleResNet50(pretrained=True, requires_grad=True, num_keypoints=num_keypoints).to(args.device)
    checkpoint = torch.load(args.model_path, map_location=torch.device(args.device), weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    #Remove mismatched output layers (due to different number of snow poles)
    state_dict.pop("l0.weight", None)
    state_dict.pop("l0.bias", None)
    model.load_state_dict(state_dict, strict=False)
    return model

# Define a function to sample every third photo
## Only used for experiments 
def sample_every_x(group, x):
    indices = np.arange(len(group[1]))
    every_x = len(group[1])//x
    selected_indices = indices[2::every_x]  
    return group[1].iloc[selected_indices]

def train_test_split(csv_path, all_images):

    df_data = pd.read_csv(csv_path)
    print(f'all rows in df_data {len(df_data.index)}')

    ## check to make sure we only use images that exist
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
    if not os.path.exists(f"{paths.get('models_output')}"):
        os.makedirs(f"{paths.get('models_output')}", exist_ok=True)
    training_samples.to_csv(f"{paths.get('models_output')}/training_samples.csv")
    validation_samples.to_csv(f"{paths.get('models_output')}/validation_samples.csv")

    print(f'# of examples we will now train on {len(training_samples)}, val on {len(validation_samples)}')
    return training_samples, validation_samples

# training function
def fit(args, model, dataloader, data, optimizer, criterion):
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

        #mask logic
        valid_mask = keypoints != -999.0
        loss = criterion(outputs[valid_mask], keypoints[valid_mask])

        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss/counter
    return train_loss

# validation function
def validate(args, model, dataloader, data, epoch, criterion):
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

            #mask logic
            valid_mask = keypoints != -999.0
            loss = criterion(outputs[valid_mask], keypoints[valid_mask]) ## cross entropy loss between input and output

            valid_running_loss += loss.item()
            # plot the predicted validation keypoints after every...
            # ... predefined number of epochs
            if not os.path.exists(args.models_output):
                os.makedirs(args.models_output, exist_ok=True)
            if (epoch + 1) % 1 == 0:
                utils.valid_keypoints_plot(args, image, outputs, keypoints, epoch)
        
    valid_loss = valid_running_loss/counter
    return valid_loss

def perform_training(args, model, train_loader, train_data, validation_loader, validation_data):
    print("Training Starting...")

    writer = SummaryWriter(f'runs/CHRL_2026')

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5, min_lr=1e-6)
    criterion = nn.SmoothL1Loss()

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
        train_epoch_loss = fit(args, model, train_loader, train_data, optimizer, criterion)
        val_epoch_loss = validate(args, model, validation_loader, validation_data, epoch, criterion)
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

    loss_plot(args, train_loss, val_loss, model, optimizer, criterion, writer)

def loss_plot(args, train_loss, val_loss, model, optimizer, criterion, writer):
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

def main():
    print("Preparing training and validation samples...")
    # Argument parser
    args = ArgumentParser("Train a model on a set of images", "train")
    matplotlib.style.use('ggplot')
    # start_time = time.time() 

    all_images = list(Path(paths.get('data_directory')).rglob("*.JPG"))
    parents_dict = {i.name: str(i) for i in all_images}

    # get the training and validation data samples
    training_samples, validation_samples = train_test_split(f"{paths.get('labels')}", all_images)

    # initialize the dataset - `snowPoleDataset()`
    train_data = SnowPoleDataset(
        training_samples,
        parents_dict,
        aug=training.get('aug'),
    )  ## we want all folders

    validation_data = SnowPoleDataset(
        validation_samples, 
        parents_dict, 
        aug=False
    )  # we always want the transform to be the normal transform

    print("Preparing data loaders")
    # # prepare data loaders
    train_loader = DataLoader(
        train_data, 
        batch_size=training.get('batch_size'), 
        shuffle=True, 
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=training.get('batch_size'),
        shuffle=False,
        num_workers=0,
    )

    print(f"Training sample instances: {len(train_data)}")
    print(f"Validation sample instances: {len(validation_data)}")

    if training.get("show_dataset_plot"):
        utils.dataset_keypoints_plot(train_data)
        utils.dataset_keypoints_plot(validation_data)

    ## create output path
    if not os.path.exists(f"{args.models_output}"):
        os.makedirs(f"{args.models_output}", exist_ok=True)

    # model
    model = load_model(args)
    print("fine-tuned model loaded...")

    perform_training(args, model, train_loader, train_data, validation_loader, validation_data)

    print("DONE TRAINING")
    # print("My program took", time.time() - start_time, "to run")

if __name__ == "__main__":
     main()