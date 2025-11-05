import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title="Image Similarity Checker", layout="centered")

st.title("🧠 Image Similarity Checker using Image Processing")

st.write("""
Upload two images below.  
The app will:
- Enhance and compress both images  
- Apply morphological operations  
- Compare them using Structural Similarity Index (SSIM)
""")

# --- Image upload ---
col1, col2 = st.columns(2)
with col1:
    img1_file = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"], key="img1")
with col2:
    img2_file = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"], key="img2")


# --- Helper functions ---
def preprocess_image(image_bytes):
    """Enhance, compress, and apply morphology to an image."""
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")

    # Resize to standard size for consistency
    img = cv2.resize(img, (400, 400))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Image Enhancement ---
    enhanced = cv2.equalizeHist(gray)

    # --- Compression (simulated by JPEG encoding/decoding) ---
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # 70% quality
    _, compressed = cv2.imencode('.jpg', enhanced, encode_param)
    compressed = cv2.imdecode(compressed, cv2.IMREAD_GRAYSCALE)

    # --- Morphological Processing ---
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(compressed, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    return morph


def compare_images(img1, img2):
    """Compute SSIM similarity between two preprocessed images."""
    score, diff = ssim(img1, img2, full=True)
    diff = (diff * 255).astype("uint8")
    return score, diff


# --- Main Logic ---
if img1_file and img2_file:
    with st.spinner("Processing images..."):
        # Preprocess both images
        img1 = preprocess_image(img1_file.read())
        img2 = preprocess_image(img2_file.read())

        # Compare
        similarity_score, diff = compare_images(img1, img2)

        st.success(f"✅ Similarity Score: **{similarity_score:.4f}**")

        if similarity_score > 0.95:
            st.write("🟢 The images are **Identical or Nearly Identical.**")
        elif similarity_score > 0.8:
            st.write("🟡 The images are **Similar but not exactly same.**")
        else:
            st.write("🔴 The images are **Different.**")

        # Display images and difference map
        st.subheader("🔍 Comparison Results")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(img1, caption="Processed Image 1", use_container_width=True)
        with c2:
            st.image(img2, caption="Processed Image 2", use_container_width=True)
        with c3:
            st.image(diff, caption="Difference Map", use_container_width=True)
else:
    st.info("Please upload two images to start comparison.")
