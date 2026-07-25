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