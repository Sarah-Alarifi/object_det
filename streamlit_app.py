import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import zipfile
import os
import torch

# Function to extract the yolov5.zip file
def extract_yolov5(zip_path, extract_to="yolov5"):
    """
    Extract the yolov5 ZIP file to the specified directory.

    Args:
        zip_path (str): Path to the yolov5 ZIP file.
        extract_to (str): Directory to extract the files into.
    """
    if not os.path.exists(extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

# Function to load the YOLO model
def load_yolo_model(model_path: str):
    """
    Load the YOLO model from the specified file path.

    Args:
        model_path (str): Path to the YOLO model file.

    Returns:
        model: The loaded YOLO model, or None if loading fails.
    """
    try:
        import sys
        sys.path.append('./yolov5')  # Add YOLOv5 to the system path
        from models.common import DetectMultiBackend
        model = DetectMultiBackend(model_path)  # Load the model
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

# Function to perform object detection
def detect_kidney_stones(image: Image.Image, model):
    """
    Detect kidney stones in the provided image using the YOLO model.

    Args:
        image (PIL.Image.Image): The input image.
        model: The YOLO model.

    Returns:
        pd.DataFrame: DataFrame containing the detection results.
    """
    try:
        # Convert PIL image to numpy array
        image_np = np.array(image)

        # Convert RGB to BGR format for YOLO
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Perform detection
        results = model(image_bgr)

        # Parse detection results
        detections = results.pandas().xyxy[0]  # Access detection results
        detections = detections[detections['name'] == 'Tas_Var']  # Filter by class name

        return detections
    except Exception as e:
        st.error(f"Error during detection: {e}")
        return pd.DataFrame()

# Streamlit UI setup
st.title("Kidney Stone Detection")
st.write("Upload an image to detect kidney stones.")

# Extract yolov5 if it hasn't been extracted yet
yolov5_zip_path = "yolov5.zip"  # Path to the ZIP file
yolov5_dir = "yolov5"  # Extraction directory
extract_yolov5(yolov5_zip_path, yolov5_dir)

# Upload the image
image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Load YOLO model
try:
    yolo_model_path = "kidney_yolo.pt"  # Replace with your YOLO model file path
    model = load_yolo_model(yolo_model_path)
    if not model:
        st.stop()
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

# Image upload and analysis
if image_file:
    # Display the uploaded image
    st.image(image_file, caption="Uploaded Image", use_column_width=True)

    # Convert to PIL Image
    image = Image.open(image_file).convert("RGB")

    # Analyze the image for kidney stones
    if st.button("Analyze for Kidney Stones"):
        detections = detect_kidney_stones(image, model)

        if not detections.empty:
            st.success("Kidney stones detected!")

            # Display detection results
            for _, row in detections.iterrows():
                st.write(f"Detected Kidney Stone - Confidence: {row['confidence']:.2f}")

            # Draw bounding boxes around detected kidney stones
            image_np = np.array(image)
            for _, row in detections.iterrows():
                x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Display the annotated image
            st.image(image_np, caption="Detected Kidney Stones", use_column_width=True)
        else:
            st.error("No kidney stones detected in the image.")
