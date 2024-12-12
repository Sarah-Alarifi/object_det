import zipfile
import os

# Ensure the yolov5 directory exists, otherwise unzip it
if not os.path.exists("yolov5"):
    with zipfile.ZipFile("yolov5.zip", 'r') as zip_ref:
        zip_ref.extractall("yolov5")
