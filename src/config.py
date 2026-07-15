import tomli as tomllib

# Read config.toml
with open("config.toml", "rb") as configfile:
    config = tomllib.load(configfile)

paths = config.get("paths", {})
cameras = config.get("cameras", {})
labeling = config.get("labeling", {})
training = config.get("training", {})