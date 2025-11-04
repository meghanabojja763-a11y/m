import streamlit as st
import cv2
import numpy as np
import os
from tempfile import TemporaryDirectory
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor

# ---------- IMAGE PROCESSING ----------
def preprocess_image(image_bytes):
    """Read and preprocess image from bytes."""
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    return enhanced

def quick_compare(img1, img2):
    """Fast SSIM similarity score."""
    h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))
    score, _ = ssim(img1_resized, img2_resized, full=True)
    return score

# ---------- MAIN SEARCH FUNCTION ----------
def search_image_in_dataset(dataset_images, search_image):
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for idx, (name, img) in enumerate(dataset_images):
            futures.append(executor.submit(quick_compare, img, search_image))

        for i, f in enumerate(futures):
            score = f.result()
            results.append((dataset_images[i][0], score))
            progress_bar.progress((i + 1) / len(dataset_images))
            status_text.text(f"Compared {i+1}/{len(dataset_images)} images...")

    progress_bar.empty()
    status_text.empty()

    best_match = max(results, key=lambda x: x[1]) if results else None
    return best_match

# ---------- STREAMLIT APP ----------
st.title("🔍 Image Similarity Search App")

st.write("""
Upload a **dataset of images** and a **search image** to find the most similar match.
You can upload **any number of images**, not just a zip file.
""")

dataset_files = st.file_uploader(
    "Upload Dataset Images (you can select multiple files)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

search_image_file = st.file_uploader(
    "Upload the Search Image",
    type=["jpg", "jpeg", "png"]
)

if dataset_files and search_image_file:
    if st.button("Search Similar Image"):
        with st.spinner("Processing... please wait"):
            dataset_images = []
            for f in dataset_files:
                try:
                    img = preprocess_image(f)
                    dataset_images.append((f.name, img))
                except Exception as e:
                    st.warning(f"Skipped {f.name}: {e}")

            search_image = preprocess_image(search_image_file)
            match = search_image_in_dataset(dataset_images, search_image)

        if match:
            best_name, score = match
            st.success(f"✅ Best match: **{best_name}** (Similarity: {score:.4f})")

            # Show side-by-side images
            col1, col2 = st.columns(2)
            with col1:
                st.image(search_image_file, caption="Search Image", use_container_width=True)
            with col2:
                matched = [f for f in dataset_files if f.name == best_name][0]
                st.image(matched, caption=f"Matched: {best_name}", use_container_width=True)
        else:
            st.error("No match found.")
else:
    st.info("👆 Upload both dataset and search image to begin.")
