import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch

# Load the YOLO model

def load_yolo_model(model_path: str):
    try:
        import sys
        sys.path.append('./yolov5')  # Add YOLOv5 to the system path
        from models.common import DetectMultiBackend
        model = DetectMultiBackend(model_path)  # Load the model
        return model
    except Exception as e:
        st.error(f"Error loading local YOLO model: {e}")
        return None

# Perform object detection using YOLO

def detect_kidney_stones(image: Image.Image, model):
    try:
        # Convert PIL image to numpy array
        image_np = np.array(image)

        # YOLO expects BGR format, so convert RGB to BGR
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Perform detection
        results = model(image_bgr)

        # Parse results
        detections = results.pandas().xyxy[0]  # Access detection results
        detections = detections[detections['name'] == 'Tas_Var']  # Filter by class name

        return detections
    except Exception as e:
        st.error(f"Error during detection: {e}")
        return pd.DataFrame()

# Streamlit UI
st.title("Kidney Stone Detection")
st.write("Upload an image to detect kidney stones.")

# File uploader for the image
image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Load the YOLO model
try:
    yolo_model_path = "kidney_yolo.pt"  # Replace with your model file path
    model = load_yolo_model(yolo_model_path)
    if not model:
        st.stop()
except FileNotFoundError as e:
    st.error(f"Missing model file: {e}")
    st.stop()

# Image upload and analysis
if image_file:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)

    image = Image.open(image_file).convert("RGB")

    # Button to trigger detection
    if st.button("Analyze for Kidney Stones"):
        detections = detect_kidney_stones(image, model)

        if not detections.empty:
            st.success("Kidney stones detected!")

            # Display detection results
            for _, row in detections.iterrows():
                st.write(f"Detected Kidney Stone - Confidence: {row['confidence']:.2f}")

            # Draw bounding boxes on the image
            image_np = np.array(image)
            for _, row in detections.iterrows():
                x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            st.image(image_np, caption="Detected Kidney Stones", use_column_width=True)
        else:
            st.error("No kidney stones detected in the image.")
