'''
written by Catherine Breen 
June 2024

If after the predictions you want to predict snow depth again
such as if you have improved metadata, you can run this script by itself on the predictions and the metadata

example command line to run:

python src/depth_conversion.py --predictions_path '/predictions/results.csv' --metadata 'example_nontrained_data/pole_metadata.csv'

'''
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance
from pathlib import Path

from arg_parser import ArgumentParser

def main():
    args = ArgumentParser("Convert pixel lengths into snow depth")

    predictions = pd.read_csv(f'{args.models_output}/predictions/results.csv')
    metadata = pd.read_csv(f"{args.path}/pole_metadata.csv")

    depth_data = {
        "camera_id": [],
        "filename": [],
    }

    for i, filename in tqdm(enumerate(predictions['filename'])):
        try: 
            camera = Path(filename).name.split("_")[0]
            
            current_idx = len(depth_data['filename'])

            depth_data["camera_id"].append(camera)
            depth_data["filename"].append(filename)

            camera_meta = metadata[metadata["camera_id"] == camera]
            img_row = predictions[predictions['filename'] == filename]

            for j in range(args.number_of_poles):
                poleId = j+1

                full_length_pole_cm = camera_meta[f's{poleId}_pole_length_cm'].values[0]
                pixel_cm_conversion = camera_meta[f's{poleId}_pixel_cm_conversion'].values[0]

                ## need to scale back up 
                x1 = img_row[f's{poleId}_x1_pred'].values[0] 
                y1 = img_row[f's{poleId}_y1_pred'].values[0] 
                x2 = img_row[f's{poleId}_x2_pred'].values[0] 
                y2 = img_row[f's{poleId}_y2_pred'].values[0]

                total_length_pixel = distance.euclidean([x1,y1],[x2,y2])
                snow_depth = full_length_pole_cm - (pixel_cm_conversion * total_length_pixel)

                # Append sideways using our matching flat structure pattern
                for key, val in [
                    (f"s{poleId}_total_length_pixel", total_length_pixel),
                    (f"s{poleId}_snowdepth_cm", snow_depth)
                ]:
                    if key not in depth_data:
                        # Pad with None for any previously processed images
                        depth_data[key] = [None] * current_idx
                    depth_data[key].append(val)

        except Exception as e:
            # If a row fails or a column doesn't exist, log it and keep moving
            print(f"Skipping image {filename} due to processing error: {e}")
            pass

    df = pd.DataFrame(depth_data)
    df.to_csv(f'{args.models_output}/predictions/results_wsnowdepthcm.csv', index=False)

    print(f'saved at {args.models_output}/predictions/results_wsnowdepthcm.csv')

if __name__ == '__main__':
    main()



