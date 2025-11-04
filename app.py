import streamlit as st
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

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
# 🗂️ Cached Dataset Loading
# -----------------------------
@st.cache_data
def load_dataset_images(dataset_folder):
    """Cache and preprocess all dataset images."""
    dataset_images = []
    for filename in os.listdir(dataset_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(dataset_folder, filename)
            try:
                dataset_images.append((filename, preprocess_image(img_path)))
            except Exception as e:
                print(f"Skipping {filename}: {e}")
    return dataset_images


# -----------------------------
# 🔍 Search Function
# -----------------------------
def search_image_in_dataset(dataset_folder, search_image_path, threshold=0.85):
    """Search dataset for similar image."""
    search_img = preprocess_image(search_image_path)
    dataset_images = load_dataset_images(dataset_folder)

    st.write(f"Found {len(dataset_images)} images in dataset.")
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Quick parallel comparison
    quick_scores = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(quick_compare, search_img, img): fname for fname, img in dataset_images}
        for i, future in enumerate(as_completed(futures)):
            filename = futures[future]
            try:
                score = future.result()
                quick_scores.append((filename, score))
            except Exception as e:
                print(f"Error on {filename}: {e}")
            progress_bar.progress((i + 1) / len(dataset_images))
            status_text.text(f"Quick comparing {filename} ({i+1}/{len(dataset_images)})")

    # Step 2: Pick top 5 for detailed comparison
    quick_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = quick_scores[:min(5, len(quick_scores))]

    st.write("### Top Quick Matches:")
    for f, s in top_candidates:
        st.write(f"**{f}** → Score: `{s:.4f}`")

    # Step 3: Detailed check
    best_match = None
    best_score = 0
    for filename, _ in top_candidates:
        dataset_img = [img for f, img in dataset_images if f == filename][0]
        score = detailed_compare(search_img, dataset_img)
        st.write(f"Detailed score for **{filename}**: {score:.4f}")
        if score > best_score:
            best_score = score
            best_match = filename

    # Step 4: Return results
    if best_score >= threshold:
        return {
            "filename": best_match,
            "path": os.path.join(dataset_folder, best_match),
            "similarity_score": best_score
        }
    else:
        return None


# -----------------------------
# 🚀 Streamlit UI
# -----------------------------
st.title("🔍 Image Similarity Search App")
st.write("Upload an image and compare it with a dataset in real time.")

dataset_folder = st.text_input("Enter the path to your dataset folder:")
search_image_file = st.file_uploader("Upload a search image", type=["jpg", "jpeg", "png"])

if dataset_folder and search_image_file:
    temp_path = os.path.join("temp_search.jpg")
    with open(temp_path, "wb") as f:
        f.write(search_image_file.getbuffer())

    st.image(temp_path, caption="Search Image", width=200)

    if st.button("Search for Similar Image"):
        with st.spinner("Searching for similar images..."):
            result = search_image_in_dataset(dataset_folder, temp_path)
        
        st.write("## 🔎 Search Results")
        if result:
            st.success(f"✅ Match Found: {result['filename']} (Score: {result['similarity_score']:.4f})")
            st.image(result["path"], caption=f"Matched Image: {result['filename']}")
        else:
            st.error("❌ No similar image found in dataset.")
