# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
from yolov11 import YOLOv11  # Replace with your YOLOv11 import
from facenet_pytorch import InceptionResnetV1
import faiss

# ----------------- Initialize Models -----------------
st.title("🟢 Face Attendance System")

@st.cache_resource
def load_models():
    yolo_model = YOLOv11()  # Load your trained YOLOv11 model
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
    # FAISS index: 512-dimensional embedding
    index = faiss.IndexFlatL2(512)
    return yolo_model, facenet_model, index

yolo_model, facenet_model, faiss_index = load_models()

# ----------------- Sidebar -----------------
st.sidebar.title("Settings")
k_neighbors = st.sidebar.slider("FAISS nearest neighbors (k)", 1, 5, 1)

# ----------------- Camera Input -----------------
st.subheader("Capture Image")
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Convert to OpenCV image
    image = Image.open(img_file_buffer)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # ----------------- YOLO Face Detection -----------------
    detections = yolo_model(frame)  # Should return list of dicts with 'bbox'

    if len(detections) == 0:
        st.warning("No faces detected!")
    else:
        st.success(f"{len(detections)} face(s) detected!")

        for det in detections:
            # YOLO bbox
            x1, y1, x2, y2 = det['bbox']
            face = frame[y1:y2, x1:x2]

            # ----------------- FaceNet Embedding -----------------
            face_resized = cv2.resize(face, (160, 160))
            face_tensor = np.transpose(face_resized, (2, 0, 1)) / 255.0
            face_tensor = np.expand_dims(face_tensor, axis=0).astype(np.float32)
            face_tensor = torch.tensor(face_tensor)

            with torch.no_grad():
                embedding = facenet_model(face_tensor).numpy()

            # ----------------- FAISS Search -----------------
            if faiss_index.ntotal > 0:
                D, I = faiss_index.search(embedding, k_neighbors)
                st.write(f"Nearest match index: {I}, distance: {D}")
            else:
                st.info("FAISS index is empty. Add embeddings first!")

            # ----------------- Draw Bounding Boxes -----------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display processed frame
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

# ----------------- Add Embeddings -----------------
st.subheader("Add Face to FAISS Index")
name = st.text_input("Name")
add_face_file = st.file_uploader("Upload face image", type=["jpg", "png"])
if add_face_file is not None and name:
    image = Image.open(add_face_file)
    face = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    face_resized = cv2.resize(face, (160, 160))
    face_tensor = np.transpose(face_resized, (2, 0, 1)) / 255.0
    face_tensor = np.expand_dims(face_tensor, axis=0).astype(np.float32)
    face_tensor = torch.tensor(face_tensor)

    with torch.no_grad():
        embedding = facenet_model(face_tensor).numpy()
        faiss_index.add(embedding)
        st.success(f"Added {name} to FAISS index!")
