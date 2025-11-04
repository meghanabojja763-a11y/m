import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import zipfile
from PIL import Image
from io import BytesIO
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor, as_completed


# ----------------------------- #
# 🧠 IMAGE PREPROCESSING
# ----------------------------- #
def preprocess_image(image_path):
    """Read and preprocess image: grayscale + enhance + resize."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    resized = cv2.resize(enhanced, (128, 128))
    return resized


# ----------------------------- #
# ⚡ IMAGE COMPARISON
# ----------------------------- #
def compare_images(img1, img2):
    """Compute SSIM similarity score between two images."""
    score, _ = ssim(img1, img2, full=True)
    return score


# ----------------------------- #
# 🗂️ LOAD DATASET (ZIP)
# ----------------------------- #
def load_dataset_from_zip(zip_bytes: bytes):
    """Extract ZIP and preprocess all images."""
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "dataset.zip")

    # Save ZIP temporarily
    with open(zip_path, "wb") as f:
        f.write(zip_bytes)

    # Extract ZIP
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    # Process dataset
    dataset_images = []
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(root, file)
                try:
                    img = preprocess_image(file_path)
                    dataset_images.append((file, img, file_path))
                except Exception as e:
                    print(f"Skipping {file}: {e}")
    return dataset_images


# ----------------------------- #
# 🔍 SEARCH FUNCTION
# ----------------------------- #
def search_image(dataset_images, search_image, threshold=0.95):
    """Compare search image with all dataset images and return best match."""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(compare_images, search_image, img): (name, path)
            for name, img, path in dataset_images
        }

        for i, future in enumerate(as_completed(futures)):
            name, path = futures[future]
            try:
                score = future.result()
                results.append((name, path, score))
            except Exception as e:
                print(f"Error comparing {name}: {e}")
            progress_bar.progress((i + 1) / len(dataset_images))
            status_text.text(f"Compared {i+1}/{len(dataset_images)} images...")

    progress_bar.empty()
    status_text.empty()

    best_match = max(results, key=lambda x: x[2]) if results else None
    if best_match and best_match[2] >= threshold:
        return best_match
    return None


# ----------------------------- #
# 🚀 STREAMLIT UI
# ----------------------------- #
st.title("🔍 Image Search in ZIP Dataset")
st.write("Upload a **ZIP folder of images** and a **search image** to check if it’s present (or similar) in the dataset.")

# Upload ZIP dataset
dataset_zip = st.file_uploader("📦 Upload Dataset (ZIP)", type=["zip"])

# Upload search image
search_image_file = st.file_uploader("🖼️ Upload Search Image", type=["jpg", "jpeg", "png"])

if dataset_zip and search_image_file:
    if st.button("Search"):
        with st.spinner("Processing dataset..."):
            # Convert to bytes (important fix for UnhashableParamError)
            zip_bytes = dataset_zip.read()
            dataset_images = load_dataset_from_zip(zip_bytes)
            st.success(f"✅ Loaded {len(dataset_images)} images from ZIP.")

        # Save and preprocess search image
        search_bytes = search_image_file.getvalue()
        search_image_path = os.path.join(tempfile.gettempdir(), "search_image.jpg")
        with open(search_image_path, "wb") as f:
            f.write(search_bytes)

        search_img = preprocess_image(search_image_path)

        with st.spinner("Searching for similar images..."):
            result = search_image(dataset_images, search_img)

        st.write("## 🔎 Search Result")
        if result:
            best_name, best_path, score = result
            st.success(f"✅ Match found: **{best_name}** (Similarity: {score:.4f})")

            col1, col2 = st.columns(2)
            with col1:
                st.image(Image.open(BytesIO(search_bytes)), caption="Search Image", use_container_width=True)
            with col2:
                st.image(Image.open(best_path), caption=f"Matched Image: {best_name}", use_container_width=True)
        else:
            st.error("❌ No exact or similar match found.")
else:
    st.info("👆 Upload a ZIP dataset and one search image to begin.")
