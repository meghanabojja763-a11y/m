import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.title("🔍 YOLOv9 Object Detection")

# Load model (auto-downloads, CPU only)
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # Use YOLOv8 (more stable than v9 on cloud)
    return model

model = load_model()

# Sidebar confidence slider
conf_threshold = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)

# Live camera
st.header("📸 Live Camera")
camera_image = st.camera_input("Take a photo")
if camera_image:
    image = Image.open(camera_image)
    st.image(image, caption="Original", use_column_width=True)
    
    # Detect
    results = model(image, conf=conf_threshold, verbose=False)
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detected Objects", use_column_width=True)
    
    # Show results
    st.subheader("Results")
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            st.write(f"**{label}**: {conf:.2f}")
    else:
        st.write("No objects detected")

# Upload image
st.header("📁 Upload Image")
uploaded_file = st.file_uploader("Choose image", type=['png','jpeg','jpg'])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original", use_column_width=True)
    
    results = model(image, conf=conf_threshold, verbose=False)
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detected Objects", use_column_width=True)
    
    st.subheader("Results")
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            st.write(f"**{label}**: {conf:.2f}")
