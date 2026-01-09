import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import torch

st.title("Live YOLOv9 Face/Object Detection")

# Load YOLOv9 model (use yolov9-c face weights or general; download auto on first run)
@st.cache_resource
def load_model():
    model = YOLO("yolov9-c.pt")  # Or face-specific: "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-face.pt"
    return model

model = load_model()

# Preprocessing function
def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (640, 640))
    image = image / 255.0  # Normalize
    return image

# Streamlit pages
tab1, tab2 = st.tabs(["Live Webcam", "Upload Image"])

with tab1:
    st.header("Live Detection")
    frame_window = st.image([])
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Preprocess
            input_frame = preprocess_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            
            # Run inference
            results = model(input_frame, conf=0.5)
            
            # Draw results
            annotated = results[0].plot()
            frame_window.image(annotated)
            
        if st.button("Stop"):
            break
    camera.release()

with tab2:
    st.header("Image Upload")
    uploaded = st.file_uploader("Choose image", type=["jpg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Original")
        
        # Preprocess and detect
        preprocessed = preprocess_image(image)
        results = model(preprocessed, conf=0.5)
        
        annotated = results[0].plot()
        st.image(annotated, caption="Detected (faces/objects)")

st.info("Detection classes: Filter for 'face' or others via model config. Outputs bounding boxes/confidence.")
