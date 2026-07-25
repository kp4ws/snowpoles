"""
Author: Kent Pawson (2026)

Used to create a clean copy of the stake_measurements.csv containing relevant columns for the project.

py src/preprocess_measurements.py
"""

import pandas as pd

def process_stake_measurements(path, file):
    measurements_df = pd.read_csv(f"{path}/{file}")

    #Preprocess data to get clean dataset
    clean_df = measurements_df[
        (measurements_df["axis"] == "slant") &
        (measurements_df["point1"] == "ground") &
        (measurements_df["point2"] == "top")
    ]

    #Group multiple rows into a total of 3 (average of all grouped rows)
    clean_df = clean_df.groupby(["site", "stake"], as_index=False)["distance_cm"].mean()

    #Optional rename columns
    clean_df = clean_df.rename(columns={
        "site": "camera_id",
        "stake": "stake_id",
        "distance_cm": "pole_length_cm"
    })

    clean_df.to_csv(f"{path}/stake_measurements_clean.csv")
    print(clean_df)

if __name__ == "__main__":
    #TODO: consider refactoring hardcoded route
    process_stake_measurements("CHRL_data", "stake_measurements.csv")