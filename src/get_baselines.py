'''
Used to retrieve baseline (snow-free) images for each camera site.
Upon selecting baseline images, they'll be copied and added to the current camera's photos and named "zz_baseline.jpg"

When selecting the baseline image in the file browser window, you can adjust the view settings in the top right and select "extra large icons" to easily see the images
'''

from pathlib import Path
import shutil
from arg_parser import ArgumentParser
from config import cameras

import tkinter as tk
from tkinter import filedialog
from pathlib import Path

def find_and_confirm_baseline_image(cam_dir, camera_id):
    '''
    Uses tkinter to open file browser and allow user to select baseline image
    '''
    # Initialize tkinter and hide the useless root window
    root = tk.Tk()
    root.withdraw()
    
    # Force the window to pop up on top of other windows
    root.attributes('-topmost', True)

    print(f"Opening file browser to select baseline image for {camera_id}...")

    # Open native file selection dialog
    file_path = filedialog.askopenfilename(
        title=f"Select Baseline Image for {camera_id}",
        initialdir=str(cam_dir),
        filetypes=[("JPEG Images", "*.jpg *.JPG"), ("All Files", "*.*")]
    )

    # Handle the case where the user clicks "Cancel"
    if not file_path:
        print("Selection cancelled. No baseline image chosen.")
        return ""

    selected_file = Path(file_path)
    print(f"--> Selected baseline: {selected_file.name}")
    
    return selected_file

def main():
    args = ArgumentParser("Select baseline images")

    data_dir = Path(args.path)
    cam_dirs = [x for x in data_dir.iterdir() if x.is_dir()]

    for cam_dir in cam_dirs:
        camera_id = cam_dir.name
        camera_cfg = cameras.get(camera_id)

        # Skip if cameraID is not in config
        if not camera_cfg:
            print(f"{cam_dir} not found in config, skipping")
            continue

        is_enabled = camera_cfg.get("enabled", True)
        active_poles = camera_cfg.get("active_poles", [])

        # Skip if camera is not enabled or has no active poles
        if not is_enabled or not active_poles:
            print(f"{cam_dir} not active, skipping")
            continue 

        #Get baseline image for current camera site
        baseline_file = cam_dir / "zz_baseline.jpg"
        #If baseline image doesn't currently exist, prompt user to select baseline image
        if not baseline_file.exists():
            try:
                #Prompt user to select baseline image
                selected_file = find_and_confirm_baseline_image(cam_dir, camera_id)
                #Copy and rename selected file to "zz_baseline.jpg" in same folder
                shutil.copy(selected_file, baseline_file)
                print(f"Saved local baseline image: {baseline_file}")
            except Exception as e:
                print(f"Error resolving baseline image: {e}. Skipping camera.")
                continue
        else:
            print(f"Baseline image already exists for {camera_id}, skipping.")

if __name__ == "__main__":
    main()