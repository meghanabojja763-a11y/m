import streamlit as st
import cv2
import numpy as np
import zipfile
import tempfile
import os
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title="Folder Image Matching", layout="centered")

st.title("🧠 Folder Image Matching using Image Processing")

st.write("""
Upload a **single image** and a **folder (ZIP file)** of images.  
The app will:
- Enhance, compress, and morphologically process all images  
- Compare the single image with each image in the folder  
- Show whether the image is present or not based on similarity
""")

# --- File Uploads ---
query_file = st.file_uploader("📸 Upload the Single Image", type=["jpg", "jpeg", "png"])
folder_zip = st.file_uploader("🗂️ Upload the Folder (as ZIP file containing images)", type=["zip"])


# --- Helper Functions ---
def preprocess_image(image_bytes):
    """Enhance, compress, and apply morphology to an image."""
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")

    # Resize for consistency
    img = cv2.resize(img, (400, 400))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhancement
    enhanced = cv2.equalizeHist(gray)

    # Compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    _, compressed = cv2.imencode('.jpg', enhanced, encode_param)
    compressed = cv2.imdecode(compressed, cv2.IMREAD_GRAYSCALE)

    # Morphological processing
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(compressed, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    return morph


def compare_images(img1, img2):
    """Compute SSIM similarity between two preprocessed images."""
    score, _ = ssim(img1, img2, full=True)
    return score


# --- Main Logic ---
if query_file and folder_zip:
    with st.spinner("Processing images..."):
        # Preprocess the single (query) image
        query_img = preprocess_image(query_file.read())

        # Create temp directory for extracted folder
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(folder_zip, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        matched_files = []
        threshold = 0.9  # similarity threshold

        # Iterate through all extracted files
        for file_name in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file_name)
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            with open(file_path, "rb") as f:
                folder_img = preprocess_image(f.read())

            score = compare_images(query_img, folder_img)

            if score >= threshold:
                matched_files.append((file_name, score))

        # --- Results ---
        if matched_files:
            st.success(f"✅ Found {len(matched_files)} matching image(s) in the folder!")
            for name, score in matched_files:
                st.write(f"🔹 **{name}** — Similarity Score: `{score:.4f}`")
                st.image(os.path.join(temp_dir, name), caption=name, use_container_width=True)
        else:
            st.error("❌ No matching image found in the folder.")

else:
    st.info("Please upload a single image and a folder (as ZIP).")
