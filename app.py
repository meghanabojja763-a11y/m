import streamlit as st
import cv2
import numpy as np
import zipfile
import tempfile
import os
from skimage.metrics import structural_similarity as ssim

# ------------------- OPTIONAL DEEP LEARNING -------------------
try:
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.models import Model
    DL_OK = True
except ImportError:
    DL_OK = False
    DL_IMPORT_ERR = "TensorFlow not installed. Deep Learning similarity disabled."
    st.warning(f"⚠️ {DL_IMPORT_ERR}")

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Advanced Image Matching", layout="centered")

st.title("🧠 Advanced Folder Image Matching using ORB + SSIM + DL (Optional)")

st.write("""
Upload a **single image** and a **folder (ZIP)** of images.  
This tool will:
- Enhance and morphologically process images  
- Extract visual features using ORB (feature matching)  
- Compare using SSIM and optionally Deep Learning (ResNet50 feature similarity)  
- Detect whether the image is present in the folder (even if resized or enhanced)
""")

# ------------------- FILE UPLOADS -------------------
query_file = st.file_uploader("📸 Upload the Single Image", type=["jpg", "jpeg", "png"])
folder_zip = st.file_uploader("🗂️ Upload the Folder (ZIP containing images)", type=["zip"])

# ------------------- HELPER FUNCTIONS -------------------
def preprocess_image(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")
    img = cv2.resize(img, (400, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    return morph

def orb_similarity(img1, img2):
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
    return max(0, min(1, 1 - avg_distance / 100))

def ssim_similarity(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    score, _ = ssim(img1, img2, full=True)
    return score

def extract_all_images(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    image_paths = []
    for root, _, files in os.walk(extract_to):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, f))
    return image_paths

# ------------------- DEEP LEARNING FUNCTIONS -------------------
if DL_OK:
    @st.cache_resource
    def load_resnet_model():
        base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        model = Model(inputs=base_model.input, outputs=base_model.output)
        return model

    def extract_resnet_features(model, img_bytes):
        file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x, verbose=0)
        return features.flatten()

    def cosine_similarity(v1, v2):
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# ------------------- MAIN LOGIC -------------------
if query_file and folder_zip:
    with st.spinner("⚙️ Processing images..."):
        query_bytes = query_file.read()
        query_img = preprocess_image(query_bytes)

        temp_dir = tempfile.mkdtemp()
        image_paths = extract_all_images(folder_zip, temp_dir)
        st.info(f"Found {len(image_paths)} images in the folder.")

        matches = []
        orb_threshold = 0.6
        ssim_threshold = 0.7
        deep_threshold = 0.85  # Only used if DL is enabled

        if DL_OK:
            resnet_model = load_resnet_model()
            query_features = extract_resnet_features(resnet_model, query_bytes)

        for path in image_paths:
            with open(path, "rb") as f:
                folder_bytes = f.read()
            folder_img = preprocess_image(folder_bytes)

            orb_score = orb_similarity(query_img, folder_img)
            ssim_score = ssim_similarity(query_img, folder_img)
            deep_score = 0.0
            if DL_OK:
                folder_features = extract_resnet_features(resnet_model, folder_bytes)
                deep_score = cosine_similarity(query_features, folder_features)

            combined = (0.4 * orb_score) + (0.3 * ssim_score) + (0.3 * deep_score if DL_OK else 0.0)

            if orb_score >= orb_threshold or ssim_score >= ssim_threshold or (DL_OK and deep_score >= deep_threshold):
                matches.append((path, orb_score, ssim_score, deep_score, combined))

        if matches:
            st.success(f"✅ Found {len(matches)} possible match(es)!")
            for path, orb_s, ssim_s, deep_s, comb in sorted(matches, key=lambda x: x[4], reverse=True):
                st.write(
                    f"**{os.path.basename(path)}** — ORB: `{orb_s:.3f}`, SSIM: `{ssim_s:.3f}`"
                    + (f", DL: `{deep_s:.3f}`" if DL_OK else "")
                    + f", Combined: `{comb:.3f}`"
                )
                st.image(path, caption=f"Matched Image (Score: {comb:.3f})", use_container_width=True)
        else:
            st.error("❌ No matching image found.")
else:
    st.info("Please upload a single image and a folder (as ZIP).")
