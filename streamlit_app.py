import streamlit as st
from PIL import Image
import torch  # If YOLO is based on PyTorch
import numpy as np

# Title of the App
st.title("YOLO Object Detection")
st.write("Upload an image and let the YOLO model detect objects in it.")

# Cache the model loading for efficiency
@st.cache_resource
def load_model():
    # Load YOLO model (ensure you replace 'model.pt' with your actual model file)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='kidney_yolo.pt', force_reload=True)
    return model

# Load the model
model = load_model()

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Detecting objects...")
    
    # Convert image to numpy
    image_np = np.array(image)
    
    # Perform detection
    results = model(image_np)
    
    # Display results
    st.image(results.render(), caption="Detection Results", use_column_width=True)
