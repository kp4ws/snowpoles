from pathlib import Path
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS
import shutil

def get_image_datetime_exif(image_path, exif_field="DateTime") -> datetime:
    """
    Extract a datetime from image EXIF metadata.

    Parameters
    ----------
    image_path : str or Path
        Path to image file.

    exif_field : str, optional
        EXIF field to read.
        Default is "DateTime".

        Common alternatives:
            - "DateTimeOriginal"
            - "DateTimeDigitized"

    Returns
    -------
    datetime.datetime
        Parsed datetime object.
    """

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist:\n{image_path}")

    image = Image.open(image_path)
    exif_data = image.getexif()
    date_string = None

    for tag_id, value in exif_data.items():
        tag = TAGS.get(tag_id, tag_id)
        if tag == exif_field:
            date_string = value
            break

    if date_string is None:
        raise ValueError(f"EXIF field '{exif_field}' not found.")

    try:
        detected_datetime = datetime.strptime(date_string, "%Y:%m:%d %H:%M:%S")

    except ValueError:
        raise ValueError(f"Could not parse EXIF datetime:\n{date_string}")

    return detected_datetime

#Extract data from trail cam images
source_dir = Path("CHRL_data/trail_cameras")
destination_dir = Path("CHRL_data/trail_cameras_processed")

#If destination directory doesn't already exist, create it
destination_dir.mkdir(exist_ok=True)

print("Processing trail camera images...")

#For each trail camera directory
for camera_dir in source_dir.iterdir():
    #If not directory, then continue
    if not camera_dir.is_dir():
        continue
    
    #Site directory name
    camera = camera_dir.name
    #Output directory name
    output_dir= destination_dir/camera
    output_dir.mkdir(exist_ok=True)

    print(f"Processing images for camera: {camera}...")

    #For each image in the site directory
    for image_path in camera_dir.rglob("*"):
        #If not file, then continue
        if not image_path.is_file():
            continue
        
        
        #Use Ben's code to grab datetime
        _datetime = get_image_datetime_exif(image_path)
        datetime_formatted = _datetime.strftime("%Y%m%d_%H")#24hr time
        #Format name to {site}_{YYYYMMDD}_{HH}
        processed_name = f"{camera}_{datetime_formatted}.jpg"

        shutil.copy2(
            image_path,
            output_dir/processed_name
        )

print("Done processing trail camera images.")