import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch  # For loading the YOLO model

# Load the YOLO model
def load_yolo_model(model_path: str):
    try:
        return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
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
        detections = results.pandas().xyxy[0]
        detections = detections[detections['name'] == 'Tas_Var']  # Filter by class name

        return detections
    except Exception as e:
        st.error(f"Error during detection: {e}")
        return pd.DataFrame()

st.title("Kidney Stone Detection")
st.write("Upload an image to detect kidney stones.")

# File uploader for the image
image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

try:
    # Path to the YOLO model file
    yolo_model_path = "kidney_yolo.pt"  # Replace with your model file path

    # Load YOLO model
    model = load_yolo_model(yolo_model_path)
    if not model:
        st.stop()
except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    st.stop()

if image_file:
    # Display uploaded image
    st.image(image_file, caption="Uploaded Image", use_column_width=True)
    
    # Convert the file to a PIL image
    image = Image.open(image_file).convert("RGB")

    # Button to trigger detection
    if st.button("Analyze for Kidney Stones"):
        detections = detect_kidney_stones(image, model)

        if not detections.empty:
            st.success("Kidney stones detected!")

            # Display detections
            for _, row in detections.iterrows():
                st.write(f"Detected Kidney Stone - Confidence: {row['confidence']:.2f}")

            # Optionally, display bounding boxes on the image
            image_np = np.array(image)
            for _, row in detections.iterrows():
                x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            st.image(image_np, caption="Detected Kidney Stones", use_column_width=True)
        else:
            st.error("No kidney stones detected in the image.")
