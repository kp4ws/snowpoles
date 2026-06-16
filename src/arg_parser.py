import argparse
import tomli as tomllib
import os

# Argument parser
class ArgumentParser():
    def __init__(self, description="Use a model to predict snow depth"):
        self.parser = argparse.ArgumentParser(description=description)
        self._add_arguments()
        self.args = self.parser.parse_args()
        self._get_arguments()

    def _add_arguments(self):
        self.parser.add_argument("--model", required=False, help="model to use")
        # self.parser.add_argument("--datapath", help="(deprecated) directory where images are located")
        self.parser.add_argument("--path", help="directory where images are located")
        self.parser.add_argument("--subset_to_label", help="label every N images")

        self.parser.add_argument("--device", required=False, help='device to use for processing ("cpu" or "cuda")')
        self.parser.add_argument("--output", required=False, help="directory in which to store marked images")
        self.parser.add_argument("--no_confirm", required=False, help="skip confirmation", action="store_true")

    def _get_arguments(self):
        # Get arguments from config file if they weren't specified  
        with open("config.toml", "rb") as configfile:
            config = tomllib.load(configfile)
            if not self.args.model:
                self.args.model = config["paths"]["models_output"]
            if not self.args.path:
                self.args.path = config["paths"]["input_images"]
            if not self.args.device:
                self.args.device = config["training"]["device"]
            if not self.args.output:
                self.args.output = config["paths"]["models_output"]
            if not self.args.subset_to_label:
                self.args.subset_to_label = config["labeling"]["subset_to_label"]
            if not self.args.output:
                self.args.output = config["paths"]["images_output"]

    def _confirm_arguments(self):
        if not self.args.no_confirm:
            print("\n\n# The following options were specified in config.toml or as arguments:\n")

            if(self.args.model.startswith("/")):
                print(f'Model to use:\n{self.args.model}\n')
            else:
                print(f"Model to use:\n{os.getcwd()}\n")
            
            if(self.args.path.startswith("/")):
                print(f"Directory where images are located:\n{self.args.path}")
            else:
                print(f"Directory where images are located:\n{os.getcwd()}\n")

            print(f"Device to use:\n{self.args.device}\n")

            if(self.args.output.startswith("/")):
                print(f"Directory where marked images will be stored:\n{self.args.output}\n")            
            else:
                print(f"Directory where marked images will be stored:\n{os.getcwd()}/{self.args.output}\n")

            print(f"Images to label:\nEvery {self.args.subset_to_label} images")
            
            confirmation = str(input("\nIs this OK? (y/n) "))
            if confirmation.lower() != "y":
                if confirmation.lower() == "n":
                    print(f"Edit the config file, located at {os.getcwd()}/config.toml, to your liking, or edit the command line arguments if they were specified, and then re-run this file.\n")
                else:
                    print("Invalid input.\n")
                quit()