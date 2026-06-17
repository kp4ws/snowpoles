#TODO: This is a temporary script used to flatten labels.csv
import pandas as pd

def flatten_labels(csv_path, output_path):
    df = pd.read_csv(csv_path)
    #Strict ordering to guarantee stake 1, 2, and 3 sequence
    df = df.sort_values(by=["camera_id", "filename", "datetime", "stake_id"])

    #Pivot metrics we need for training and validation
    df_pivoted = df.pivot(
        index=['camera_id', 'filename', 'datetime'],
        columns='stake_id',
        values=['x1', 'y1', 'x2', 'y2', 'pixel_length', 'snow_depth']
    )

    #Flatten column names
    df_pivoted.columns = [f's{stake}_{col}' for col, stake in df_pivoted.columns]
    df_pivoted = df_pivoted.reset_index()

    df_pivoted.to_csv(output_path, index=False)
    print("Data flattened")

if __name__ == "__main__":
    flatten_labels("CHRL_data/labels.csv", "CHRL_data/labels_flattened.csv")