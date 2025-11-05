import streamlit as st
import cv2
import numpy as np
import zipfile
import tempfile
import os
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title="Advanced Image Matching", layout="centered")

st.title("🧠 Advanced Folder Image Matching using Image Processing + ORB Features")

st.write("""
Upload a **single image** and a **folder (ZIP)** of images.  
This tool will:
- Enhance and morphologically process images  
- Extract visual features using ORB (feature matching)  
- Compare using both Feature Matching & SSIM  
- Find whether the image is present in the folder (even if resized or enhanced)
""")

# --- File Uploads ---
query_file = st.file_uploader("📸 Upload the Single Image", type=["jpg", "jpeg", "png"])
folder_zip = st.file_uploader("🗂️ Upload the Folder (as ZIP file containing images)", type=["zip"])


# --- Helper Functions ---
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

    # --- Enhancement: Histogram Equalization ---
    enhanced = cv2.equalizeHist(gray)

    # --- Morphological Operations ---
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    return morph


def orb_similarity(img1, img2):
    """Compute ORB feature matching score."""
    orb = cv2.ORB_create(1000)

    # Detect and compute descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    # Match features using Brute-Force Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) == 0:
        return 0

    # Compute similarity based on distance
    distances = [m.distance for m in matches]
    avg_distance = sum(distances) / len(distances)

    # Normalize score: smaller distance = better match
    similarity = max(0, min(1, 1 - avg_distance / 100))
    return similarity


def ssim_similarity(img1, img2):
    """Compute SSIM similarity for verification."""
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    score, _ = ssim(img1, img2, full=True)
    return score


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


# --- Main Logic ---
if query_file and folder_zip:
    with st.spinner("Processing images..."):
        query_img = preprocess_image(query_file.read())

        temp_dir = tempfile.mkdtemp()
        image_paths = extract_all_images(folder_zip, temp_dir)

        st.info(f"Found {len(image_paths)} images in the folder.")

        matches = []
        all_results = []

        orb_threshold = 0.6  # Feature-based similarity
        ssim_threshold = 0.7  # Pixel-based similarity

        for path in image_paths:
            with open(path, "rb") as f:
                folder_img = preprocess_image(f.read())

            orb_score = orb_similarity(query_img, folder_img)
            ssim_score = ssim_similarity(query_img, folder_img)
            combined = (orb_score * 0.7) + (ssim_score * 0.3)

            all_results.append((path, orb_score, ssim_score, combined))

            if orb_score >= orb_threshold or ssim_score >= ssim_threshold:
                matches.append((path, orb_score, ssim_score, combined))

        # --- Display Results ---
        if matches:
            st.success(f"✅ Found {len(matches)} possible match(es)!")
            for path, orb_s, ssim_s, comb in sorted(matches, key=lambda x: x[3], reverse=True):
                st.write(f"**{os.path.basename(path)}** — ORB: `{orb_s:.3f}`, SSIM: `{ssim_s:.3f}`, Combined: `{comb:.3f}`")
                st.image(path, caption=f"Matched Image (Score: {comb:.3f})", use_container_width=True)
        else:
            st.error("❌ No matching image found.")
            st.subheader("🔍 Top 3 Closest Images (for reference):")
            for path, orb_s, ssim_s, comb in sorted(all_results, key=lambda x: x[3], reverse=True)[:3]:
                st.write(f"**{os.path.basename(path)}** — ORB: `{orb_s:.3f}`, SSIM: `{ssim_s:.3f}`, Combined: `{comb:.3f}`")
                st.image(path, caption=f"Top Similar Image (Score: {comb:.3f})", use_container_width=True)
else:
    st.info("Please upload a single image and a folder (as ZIP).")
