import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import zipfile
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor, as_completed


# -----------------------------
# 🧠 Image Preprocessing
# -----------------------------
def preprocess_image(image_path, size=(128, 128)):
    """Read and preprocess image: enhance, resize, and apply morphology."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    resized = cv2.resize(denoised, size, interpolation=cv2.INTER_AREA)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel)
    return morph


# -----------------------------
# ⚡ Comparison Functions
# -----------------------------
def quick_compare(img1, img2):
    """Fast comparison using SSIM and histogram."""
    ssim_score, _ = ssim(img1, img2, full=False)
    hist1 = cv2.calcHist([img1], [0], None, [64], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [64], [0, 256])
    hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return 0.7 * ssim_score + 0.3 * hist_score


def detailed_compare(img1, img2):
    """More accurate but slower comparison including ORB features."""
    ssim_score, _ = ssim(img1, img2, full=True)
    hist1 = cv2.calcHist([img1], [0], None, [64], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [64], [0, 256])
    hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        match_score = len(matches) / max(len(kp1), len(kp2))
    else:
        match_score = 0
    return 0.5 * ssim_score + 0.3 * hist_score + 0.2 * match_score


# -----------------------------
# 🗂️ Dataset Loader
# -----------------------------
@st.cache_data
def load_dataset(dataset_source, is_zip):
    """Load dataset either from ZIP or from a local folder."""
    temp_dir = tempfile.mkdtemp()

    # --- Case 1: ZIP upload ---
    if is_zip:
        zip_path = os.path.join(temp_dir, "dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(dataset_source.getbuffer())
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        base_dir = temp_dir
    # --- Case 2: Local folder ---
    else:
        base_dir = dataset_source

    dataset_images = []
    for root, _, files in os.walk(base_dir):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, filename)
                try:
                    dataset_images.append((filename, preprocess_image(img_path), img_path))
                except Exception as e:
                    print(f"Skipping {filename}: {e}")
    return dataset_images


# -----------------------------
# 🔍 Search Function
# -----------------------------
def search_image_in_dataset(dataset_images, search_image_path, threshold=0.85):
    """Search dataset for similar image."""
    search_img = preprocess_image(search_image_path)

    st.write(f"Found {len(dataset_images)} images in dataset.")
    progress_bar = st.progress(0)
    status_text = st.empty()

    quick_scores = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(quick_compare, search_img, img): (fname, path)
                   for fname, img, path in dataset_images}
        for i, future in enumerate(as_completed(futures)):
            filename, path = futures[future]
            try:
                score = future.result()
                quick_scores.append((filename, path, score))
            except Exception as e:
                print(f"Error on {filename}: {e}")
            progress_bar.progress((i + 1) / len(dataset_images))
            status_text.text(f"Comparing {filename} ({i+1}/{len(dataset_images)})")

    quick_scores.sort(key=lambda x: x[2], reverse=True)
    top_candidates = quick_scores[:min(5, len(quick_scores))]

    st.write("### Top Quick Matches:")
    for f, _, s in top_candidates:
        st.write(f"**{f}** → Score: `{s:.4f}`")

    best_match = None
    best_score = 0
    for filename, path, _ in top_candidates:
        dataset_img = [img for f, img, p in dataset_images if f == filename][0]
        score = detailed_compare(search_img, dataset_img)
        st.write(f"Detailed score for **{filename}**: {score:.4f}")
        if score > best_score:
            best_score = score
            best_match = (filename, path)

    if best_score >= threshold:
        return {"filename": best_match[0], "path": best_match[1], "similarity_score": best_score}
    else:
        return None


# -----------------------------
# 🚀 Streamlit UI
# -----------------------------
st.title("🔍 Universal Image Similarity Search App")
st.write("Upload a dataset (ZIP or folder) and an image to find similar matches.")

# Mode selection
mode = st.radio("Select dataset input method:", ("Upload ZIP file", "Use Local Folder"))

dataset_images = None

# --- Option 1: Upload ZIP file ---
if mode == "Upload ZIP file":
    dataset_zip = st.file_uploader("📦 Upload Dataset (ZIP)", type=["zip"])
    if dataset_zip:
        with st.spinner("Extracting and preprocessing dataset..."):
            dataset_images = load_dataset(dataset_zip, is_zip=True)
        st.success(f"Loaded {len(dataset_images)} images from uploaded ZIP.")

# --- Option 2: Use Local Folder ---
elif mode == "Use Local Folder":
    dataset_folder = st.text_input("📁 Enter path to local dataset folder:")
    if dataset_folder and os.path.isdir(dataset_folder):
        with st.spinner("Loading dataset from local folder..."):
            dataset_images = load_dataset(dataset_folder, is_zip=False)
        st.success(f"Loaded {len(dataset_images)} images from local folder.")

# --- Upload search image ---
search_image_file = st.file_uploader("🖼️ Upload Search Image", type=["jpg", "jpeg", "png"])

if dataset_images and search_image_file:
    # Save search image temporarily
    temp_search_path = os.path.join(tempfile.gettempdir(), "temp_search.jpg")
    with open(temp_search_path, "wb") as f:
        f.write(search_image_file.getbuffer())

    st.image(temp_search_path, caption="Search Image", width=200)

    if st.button("Search for Similar Image"):
        with st.spinner("Searching for similar images..."):
            result = search_image_in_dataset(dataset_images, temp_search_path)
        
        st.write("## 🔎 Search Results")
        if result:
            st.success(f"✅ Match Found: {result['filename']} (Score: {result['similarity_score']:.4f})")
            st.image(result["path"], caption=f"Matched Image: {result['filename']}")
        else:
            st.error("❌ No similar image found in dataset.")
else:
    st.info("Please upload or select a dataset and upload a search image.")
