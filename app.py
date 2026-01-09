import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.title("🖼️ YOLOv9 Live Face/Object Detection")

@st.cache_resource
def load_model():
    """Load YOLOv9 model (auto-downloads on first run)."""
    model = YOLO("yolov9-c.pt")  # General; swap to "yolov9-c-face.pt" for faces
    model.to('cpu')  # Force CPU for Streamlit Cloud
    return model

model = load_model()

@st.cache_data
def preprocess_image(image):
    """Resize to 640x640 and normalize."""
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (640, 640))
    return img_resized.astype(np.float32) / 255.0

# Live camera input (works seamlessly on deployment)
st.header("📹 Live Webcam Detection")
camera_img = st.camera_input("Capture live image")
if camera_img:
    image = Image.open(camera_img)
    st.image(image, caption="Original Image", use_column_width=True)
    
    # Preprocess and detect
    preprocessed = preprocess_image(image)
    with st.spinner("Detecting..."):
        results = model(preprocessed, conf=0.5, device='cpu', verbose=False)
    
    annotated = results[0].plot()
    st.image(annotated, caption="Detected Objects (e.g., faces, people)", use_column_width=True)
    
    # Print results
    st.subheader("Detection Output")
    boxes = results[0].boxes
    if boxes is not None:
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            st.write(f"{label}: {conf:.2f} confidence")
    else:
        st.write("No detections found.")

# Image upload fallback
st.header("📁 Upload Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    preprocessed = preprocess_image(image)
    with st.spinner("Detecting..."):
        results = model(preprocessed, conf=0.5, device='cpu', verbose=False)
    
    annotated = results[0].plot()
    st.image(annotated, caption="Detected Objects", use_column_width=True)
    
    # Print results
    st.subheader("Detection Output")
    boxes = results[0].boxes
    if boxes is not None:
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            st.write(f"{label}: {conf:.2f} confidence")
    else:
        st.write("No detections found.")

st.info("💡 Tips: First run downloads model (~50MB). Use 'face' models from Ultralytics for better faces. Refresh for new detections.")
