import streamlit as st
import cv2
import numpy as np
import zipfile
import tempfile
import os
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Advanced Image Matching (ORB + SSIM + CNN)", layout="centered")

st.title("🧠 Advanced Folder Image Matching using Image Processing + ORB + CNN")

st.write("""
Upload a **single image** and a **folder (ZIP)** of images.

This tool combines:
- Image enhancement & morphological processing  
- ORB feature matching  
- SSIM pixel similarity  
- CNN (VGG16) deep feature similarity  
- Finds whether the image exists in the folder (even if resized, rotated, or enhanced)
""")

# ------------------- FILE UPLOADS -------------------
query_file = st.file_uploader("📸 Upload the Single Image", type=["jpg", "jpeg", "png"])
folder_zip = st.file_uploader("🗂️ Upload the Folder (as ZIP file containing images)", type=["zip"])

# ------------------- HELPER FUNCTIONS -------------------
def preprocess_image(image_bytes):
    """Enhance, morphologically filter, and prepare an image for matching."""
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")

    # Resize for uniform comparison
    img = cv2.resize(img, (400, 400))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhancement: Histogram Equalization
    enhanced = cv2.equalizeHist(gray)

    # Morphological Operations
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    return morph, img  # return both grayscale (for ORB/SSIM) and color (for CNN)

def orb_similarity(img1, img2):
    """Compute ORB feature matching score."""
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) == 0:
        return 0

    distances = [m.distance for m in matches]
    avg_distance = sum(distances) / len(distances)

    similarity = max(0, min(1, 1 - avg_distance / 100))
    return similarity

def ssim_similarity(img1, img2):
    """Compute SSIM similarity."""
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    score, _ = ssim(img1, img2, full=True)
    return score

# ------------------- CNN FEATURE EXTRACTION -------------------
@st.cache_resource
def load_vgg16_model():
    """Load pre-trained VGG16 (feature extractor)."""
    base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

def extract_cnn_features(model, img):
    """Extract deep feature vector from image using VGG16."""
    img = cv2.resize(img, (224, 224))  # required input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array, verbose=0)
    return features.flatten()

def cnn_similarity(model, img1, img2):
    """Compute deep learning similarity using cosine similarity."""
    f1 = extract_cnn_features(model, img1)
    f2 = extract_cnn_features(model, img2)
    sim = cosine_similarity([f1], [f2])[0][0]  # cosine similarity
    return float(sim)

def extract_all_images(zip_path, extract_to):
    """Extract all images (including nested folders) from ZIP."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    image_paths = []
    for root, _, files in os.walk(extract_to):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, f))
    return image_paths

# ------------------- MAIN LOGIC -------------------
if query_file and folder_zip:
    with st.spinner("Processing images..."):
        query_gray, query_color = preprocess_image(query_file.read())

        temp_dir = tempfile.mkdtemp()
        image_paths = extract_all_images(folder_zip, temp_dir)

        st.info(f"Found {len(image_paths)} images in the folder.")
        model = load_vgg16_model()

        matches = []
        all_results = []

        orb_threshold = 0.6
        ssim_threshold = 0.7
        cnn_threshold = 0.75  # deep similarity threshold

        for path in image_paths:
            with open(path, "rb") as f:
                folder_gray, folder_color = preprocess_image(f.read())

            orb_score = orb_similarity(query_gray, folder_gray)
            ssim_score = ssim_similarity(query_gray, folder_gray)
            cnn_score = cnn_similarity(model, query_color, folder_color)

            # Combined weighted score
            combined = (0.4 * orb_score) + (0.2 * ssim_score) + (0.4 * cnn_score)

            all_results.append((path, orb_score, ssim_score, cnn_score, combined))

            if cnn_score >= cnn_threshold or orb_score >= orb_threshold or ssim_score >= ssim_threshold:
                matches.append((path, orb_score, ssim_score, cnn_score, combined))

        # --- Display Results ---
        if matches:
            st.success(f"✅ Found {len(matches)} possible match(es)!")

            for path, orb_s, ssim_s, cnn_s, comb in sorted(matches, key=lambda x: x[4], reverse=True):
                st.write(f"**{os.path.basename(path)}** — ORB: `{orb_s:.3f}`, SSIM: `{ssim_s:.3f}`, CNN: `{cnn_s:.3f}`, Combined: `{comb:.3f}`")
                st.image(path, caption=f"Matched Image (Score: {comb:.3f})", use_container_width=True)
        else:
            st.error("❌ No matching image found.")
else:
    st.info("Please upload a single image and a folder (as ZIP).")
