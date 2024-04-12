
#1.Import Necessary Libraries

import streamlit as st  # Framework for building web apps
import PIL  # Python Imaging Library for image processing
from PIL import Image  # Import Image class specifically
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting images

from pathlib import Path  # For handling file paths
import torch  # PyTorch library for deep learning
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights  # Object detection model
from torchvision.utils import draw_bounding_boxes  # To draw bounding boxes on images

#2. Load Local Modules:
#These modules contain format_file and helper functions
import action_file
import format_file

# Create a function to load the model first
import sys
import os


#3. Create Containers for Content:
main_container = st.container()  # Container for main app content
container = st.sidebar.container()  # Container for sidebar elements
container.empty()  # Clear the sidebar container

#4. Title and Image Upload:
with main_container:
    st.title("Objects Detector by Andhika Nugraha :student:")
    upload_image = st.file_uploader(label="Upload Your Image & Detect Here :", type=["png", "jpg", "jpeg"], key="image_uploader")
    upload_video = st.file_uploader(label="Upload Your Video & Detect Here :", type=["mp4"], key="video_uploader")

#5.Sidebar Elements
st.sidebar.header("Image & Video Config")

# Confidence slider
numb_confidence = st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40, format="%d") / 100

# Task selection
selected_task = st.sidebar.radio(
    "Select Task",
    ['Detection', 'Segmentation'],
)

#6.Load Pre Trained Model
# Select model path based on task
if selected_task == 'Detection':
    path_model = Path(format_file.DETECTION_MODEL)
elif selected_task == 'Segmentation':
    path_model = Path(format_file.SEGMENTATION_MODEL)

# Load the model with error handling
try:
    model = action_file.load_model(path_model)
    
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {path_model}")
    st.error(ex)

#7.Image Processing Logic
if upload_image and upload_image.type.endswith(('jpg', 'png', 'jpeg')) :
    img = Image.open(upload_image).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        try:
            if img is None:
                pass
            else:
                uploaded_image = img
                st.image(img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if img is None:
            pass
        else:
            if upload_image and upload_image.type.endswith(('jpg', 'png', 'jpeg')):
                uploaded_image = img
                res = model.predict(uploaded_image, conf=numb_confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)

                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("No image is uploaded yet!")

else:
    st.error("Please select a valid source type!")

# Call the play_stored_video function with the upload_video variable
if upload_video is not None:
    action_file.play_stored_videos(numb_confidence, model, upload_video)