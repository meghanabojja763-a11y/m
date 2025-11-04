import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from skimage.metrics import structural_similarity as ssim

# -------------------------------
# Image Preprocessing Function
# -------------------------------
def preprocess_image(image_path):
    """Read and preprocess image: enhance, compress, and apply morphology."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or unreadable: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    resized = cv2.resize(denoised, (256, 256), interpolation=cv2.INTER_AREA)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel)
    return morph


# -------------------------------
# Comparison Function
# -------------------------------
def compare_images(img1_path, img2_path):
    img1 = preprocess_image(img1_path)
    img2 = preprocess_image(img2_path)

    # Structural Similarity (SSIM)
    ssim_score, _ = ssim(img1, img2, full=True)

    # Histogram Correlation
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    # ORB Feature Matching
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        match_score = len(matches) / max(len(kp1), len(kp2))
    else:
        match_score = 0.0

    # Weighted final score
    final_score = (0.5 * ssim_score) + (0.3 * hist_score) + (0.2 * match_score)
    return ssim_score, hist_score, match_score, final_score


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🖼️ Image Similarity Checker using Image Processing")
st.write("Upload a **dataset of images** and a **search image** to check which one is most similar.")

# Upload dataset
dataset_files = st.file_uploader("Upload multiple dataset images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

# Upload search image
search_image = st.file_uploader("Upload the image to search for", type=["jpg", "jpeg", "png"])

# Process if files uploaded
if dataset_files and search_image:
    st.image(search_image, caption="Search Image", use_container_width=True)

    # Save search image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_search:
        temp_search.write(search_image.read())
        search_path = temp_search.name

    results = []
    progress = st.progress(0)
    total = len(dataset_files)

    for i, dataset_file in enumerate(dataset_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_dataset:
            temp_dataset.write(dataset_file.read())
            dataset_path = temp_dataset.name

        try:
            ssim_score, hist_score, match_score, final_score = compare_images(search_path, dataset_path)
            results.append({
                "filename": dataset_file.name,
                "ssim": ssim_score,
                "hist": hist_score,
                "match": match_score,
                "score": final_score
            })
        except Exception as e:
            st.error(f"Error comparing {dataset_file.name}: {e}")

        progress.progress((i + 1) / total)

    # Sort by final similarity score
    results.sort(key=lambda x: x["score"], reverse=True)

    st.subheader("🔍 Similarity Results")
    for r in results:
        if r["score"] > 0.85:
            similarity = "✅ Likely SAME or very similar"
        elif r["score"] > 0.6:
            similarity = "🟡 Moderately similar"
        else:
            similarity = "❌ Different"

        st.write(f"**{r['filename']}** — SSIM: {r['ssim']:.3f}, Hist: {r['hist']:.3f}, ORB: {r['match']:.3f}, "
                 f"**Final Score:** {r['score']:.3f} → {similarity}")

else:
    st.info("Please upload both dataset images and a search image to start comparison.")
