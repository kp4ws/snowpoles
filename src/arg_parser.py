import argparse
import tomli as tomllib
import os

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
        self.parser.add_argument("--model", required=False, help="model to use")
        # self.parser.add_argument("--datapath", help="(deprecated) directory where images are located")
        self.parser.add_argument("--path", help="directory where images are located")
        self.parser.add_argument("--subset_to_label", help="label every N images")
        self.parser.add_argument("--device", required=False, help='device to use for processing ("cpu" or "cuda")')
        self.parser.add_argument("--models_output", required=False, help="directory in which to store models output")
        self.parser.add_argument("--images_output", required=False, help="directory in which to store images output")
        self.parser.add_argument("--no_confirm", required=False, help="skip confirmation", action="store_true")
        self.parser.add_argument("--epochs", required=False, help="epochs")
        self.parser.add_argument("--lr", required=False, help="the learning rate of the model")


    def _get_arguments(self):
        # Get arguments from config file if they weren't specified  
        with open("config.toml", "rb") as configfile:
            config = tomllib.load(configfile)
            if not self.args.model:
                self.args.model = config["paths"]["trainee_model"]
            if not self.args.path:
                self.args.path = config["paths"]["input_images"]
            if not self.args.device:
                self.args.device = config["training"]["device"]
            if not self.args.subset_to_label:
                self.args.subset_to_label = config["labeling"]["subset_to_label"]
            if not self.args.models_output:
                self.args.models_output = config["paths"]["models_output"]
            if not self.args.images_output:
                self.args.images_output = config["paths"]["images_output"]
            if not self.args.epochs:
                self.args.epochs = config["training"]["epochs"]
            if not self.args.lr:
                self.args.lr = config["training"]["lr"]

    def _confirm_arguments(self):
        if not self.args.no_confirm:
            print("\n\n# The following options were specified in config.toml or as arguments:\n")

            #Can be added as condition to print only relevant fields
            def is_relevant(field_name):
                return self.job_fields is None or field_name in self.job_fields
            
            if(self.args.model.startswith("/")):
                print(f'Model to use:\n{self.args.model}\n')
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
                print(f"Images to label:\nEvery {self.args.subset_to_label} images")
            
            if(is_relevant("train")):
                print("LR:\n" + str(self.args.lr) + "\n")
                print("Epochs:\n" + str(self.args.epochs) + "\n")

            confirmation = str(input("\nIs this OK? (y/n) "))
            if confirmation.lower() != "y":
                if confirmation.lower() == "n":
                    print(f"Edit the config file, located at {os.getcwd()}/config.toml, to your liking, or edit the command line arguments if they were specified, and then re-run this file.\n")
                else:
                    print("Invalid input.\n")
                quit()