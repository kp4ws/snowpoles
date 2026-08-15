"""
Author: Kent Pawson (2026)

Used to faciliate access to the config.toml file. 
- Communicates directly with the configuration file
- Other scripts can import the relevant config dictionaries from this script
"""

import tomli as tomllib

# Read config.toml
with open("config.toml", "rb") as configfile:
    config = tomllib.load(configfile)

paths = config.get("paths", {})
cameras = config.get("cameras", {})
labeling = config.get("labeling", {})
training = config.get("training", {})

#Determine the max number of poles of the trail cam data
global_max_poles = 0
for cam_name, cam_settings in cameras.items():
    if cam_settings.get("enabled", False):
        active_pole_count = len(cam_settings.get("active_poles", []))

        if active_pole_count > global_max_poles:
            global_max_poles = active_pole_count