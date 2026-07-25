"""
Author: Kent Pawson 2026

Contains ArgumentParser, a wrapper class for parsing arguments and providing them to be used in the various scripts
Communicates with config.py to retrieve arguments from configuration file.

Recommended use case is to specify all arguments in config.toml so you don't have to manually enter them when you run the programs in this project.
"""

import argparse
import os
from config import paths, labeling, training

# Argument parser
class ArgumentParser():
    def __init__(self, description="", job_fields=None):
        self.parser = argparse.ArgumentParser(description=description)
        self.job_fields = job_fields
        self._add_arguments()
        self.args = self.parser.parse_args()
        self._get_arguments()
        self._confirm_arguments()

    #Automatically fetch things from args by name
    def __getattr__(self, name):
        return getattr(self.args, name)

    def _add_arguments(self):
        self.parser.add_argument("--model_path", required=False, help="model to use")
        # self.parser.add_argument("--datapath", help="(deprecated) directory where images are located")
        self.parser.add_argument("--path", help="directory where images are located")
        self.parser.add_argument("--target_label_count", help="label N images")
        self.parser.add_argument("--device", required=False, help='device to use for processing ("cpu" or "cuda")')
        self.parser.add_argument("--models_output", required=False, help="directory in which to store models output")
        self.parser.add_argument("--images_output", required=False, help="directory in which to store images output")
        self.parser.add_argument("--no_confirm", required=False, help="skip confirmation", action="store_true")
        self.parser.add_argument("--epochs", required=False, help="epochs")
        self.parser.add_argument("--lr", required=False, help="the learning rate of the model")


    def _get_arguments(self):
        # Get arguments from config if they weren't specified
        if not self.args.model_path:
            models_output_dir = paths.get("models_output", "output/models")
            default_trained_path = os.path.normpath(os.path.join(models_output_dir, paths.get("trained_model_name", "model.pth")))
            
            # Check if fine-tuned model actually exists from a completed training run
            if os.path.exists(default_trained_path):
                self.args.model_path = default_trained_path
            else:
                # Fallback to base pretrained model if no trained model is found yet
                self.args.model_path = paths.get("pretrained_model", "").strip()

        if not self.args.path:
            self.args.path = paths.get("input_images")
        if not self.args.device:
            self.args.device = training.get("device", "cpu")
        if not self.args.target_label_count:
            self.args.target_label_count = labeling.get("target_label_count")
        if not self.args.models_output:
            self.args.models_output = paths.get("models_output")
        if not self.args.images_output:
            self.args.images_output = paths.get("images_output")
        if not self.args.epochs:
            self.args.epochs = training.get("epochs")
        if not self.args.lr:
            self.args.lr = training.get("lr")

    def _confirm_arguments(self):
        if not self.args.no_confirm:
            print("\n\n# The following options were specified in config.toml or as arguments:\n")

            #NOTE: Add this as condition to print only relevant fields
            def is_relevant(field_name):
                if not self.job_fields:
                    return False
                return field_name in self.job_fields
            
            if(self.args.model_path.startswith("/")):
                print(f'Model to use:\n{self.args.model_path}\n')
            else:
                print(f"Model to use:\n{os.getcwd()}\n")
            
            if(self.args.path.startswith("/")):
                print(f"Directory where images are located:\n{self.args.path}")
            else:
                print(f"Directory where images are located:\n{os.getcwd()}\n")

            print(f"Device to use:\n{self.args.device}\n")

            if(self.args.models_output.startswith("/")):
                print(f"Directory where models output will be stored:\n{self.args.models_output}\n")            
            else:
                print(f"Directory where models output will be stored:\n{os.getcwd()}/{self.args.models_output}\n")

            if(self.args.images_output.startswith("/")):
                print(f"Directory where images output will be stored:\n{self.args.images_output}\n")            
            else:
                print(f"Directory where images output will be stored:\n{os.getcwd()}/{self.args.images_output}\n")

            if(is_relevant("label")):
                print(f"Images to label:\n{self.args.target_label_count} images\n")
            
            if(is_relevant("train")):
                print("Learning Rate:\n" + str(self.args.lr) + "\n")
                print("Epochs:\n" + str(self.args.epochs) + "\n")

            if(is_relevant("label") or is_relevant("train")):
                confirmation = str(input("\nIs this OK? (y/n) "))
                if confirmation.lower() != "y":
                    if confirmation.lower() == "n":
                        print(f"Edit the config file, located at {os.getcwd()}/config.toml, to your liking, or edit the command line arguments if they were specified, and then re-run this file.\n")
                    else:
                        print("Invalid input.\n")
                    quit()