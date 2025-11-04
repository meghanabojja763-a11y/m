import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import zipfile
import urllib.request
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor, as_completed


# ----------------------------- #
# 🧠 IMAGE PREPROCESSING
# ----------------------------- #
def preprocess_image(image_path):
    """Read and preprocess image: grayscale + enhance."""
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
# 🗂️ LOAD DATASET
# ----------------------------- #
@st.cache_data(show_spinner=False)
def load_dataset_from_zip(zip_path):
    """Extract ZIP and preprocess all images."""
    temp_dir = tempfile.mkdtemp()

    # If zip_path is a file-like object
    if hasattr(zip_path, "getbuffer"):
        with open(os.path.join(temp_dir, "dataset.zip"), "wb") as f:
            f.write(zip_path.getbuffer())
        extract_path = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(os.path.join(temp_dir, "dataset.zip"), "r") as zip_ref:
            zip_ref.extractall(extract_path)
    else:
        # Local or downloaded ZIP
        extract_path = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

    dataset_images = []
    image_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(extract_path)
        for file in files if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    progress = st.progress(0)
    for i, file_path in enumerate(image_files):
        try:
            img = preprocess_image(file_path)
            dataset_images.append((os.path.basename(file_path), img, file_path))
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
        progress.progress((i + 1) / len(image_files))

    progress.empty()
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
st.title("🔍 Large ZIP Image Search App")
st.write("""
Upload or link a **large ZIP dataset of images** and a **search image**.  
The app will check if the image is present (or similar) in the dataset.
""")

# --- Option for dataset input ---
mode = st.radio(
    "Select dataset input method:",
    ["📦 Upload ZIP file", "🌐 Google Drive / URL link", "💻 Local ZIP path"]
)

zip_source = None

if mode == "📦 Upload ZIP file":
    zip_source = st.file_uploader("Upload ZIP Dataset", type=["zip"])

elif mode == "🌐 Google Drive / URL link":
    url = st.text_input("Enter ZIP file link (Google Drive / direct URL):")
    if url:
        with st.spinner("Downloading ZIP file..."):
            temp_zip = os.path.join(tempfile.gettempdir(), "dataset_download.zip")
            urllib.request.urlretrieve(url, temp_zip)
            zip_source = temp_zip
        st.success("✅ Downloaded ZIP from URL.")

elif mode == "💻 Local ZIP path":
    local_path = st.text_input("Enter local ZIP file path:")
    if local_path and os.path.exists(local_path):
        zip_source = local_path
        st.success("✅ Using local ZIP file.")

# Upload search image
search_image_file = st.file_uploader("🖼️ Upload Search Image", type=["jpg", "jpeg", "png"])

if zip_source and search_image_file:
    if st.button("Search"):
        with st.spinner("Processing dataset..."):
            dataset_images = load_dataset_from_zip(zip_source)
            st.success(f"✅ Loaded {len(dataset_images)} images from dataset.")

        # Save search image temporarily
        temp_search_path = os.path.join(tempfile.gettempdir(), "search_image.jpg")
        with open(temp_search_path, "wb") as f:
            f.write(search_image_file.getbuffer())

        search_img = preprocess_image(temp_search_path)

        with st.spinner("Searching for similar images..."):
            result = search_image(dataset_images, search_img)

        st.write("## 🔎 Search Result")
        if result:
            best_name, best_path, score = result
            st.success(f"✅ Match found: **{best_name}** (Similarity: {score:.4f})")

            col1, col2 = st.columns(2)
            with col1:
                st.image(search_image_file, caption="Search Image", use_container_width=True)
            with col2:
                st.image(best_path, caption=f"Matched Image: {best_name}", use_container_width=True)
        else:
            st.error("❌ No exact or similar match found.")
else:
    st.info("👆 Upload or link a ZIP dataset and one search image to begin.")
