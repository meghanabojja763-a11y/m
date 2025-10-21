import streamlit as st
import cv2
import numpy as np
from skimage import measure

st.set_page_config(page_title="KUB Kidney Stone Detection", layout="wide")
st.title("Kidney Stone Detection in KUB X-ray")

# Initialize session_state variables
for key in ["image", "preprocessed", "enhanced", "morph", "detected"]:
    if key not in st.session_state:
        st.session_state[key] = None

uploaded_file = st.file_uploader("Upload your KUB X-ray image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    st.session_state.image = image
    st.image(image, caption="Original X-ray", use_column_width=True)

# Preprocessing: CLAHE + Median Blur
if st.button("Preprocess Image"):
    if st.session_state.image is not None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(st.session_state.image)
        filtered = cv2.medianBlur(enhanced, 5)
        st.session_state.preprocessed = filtered
        st.image(filtered, caption="Preprocessed Image", use_column_width=True)
    else:
        st.warning("Please upload an image first.")

# Enhancement: Unsharp Masking
if st.button("Enhance Image (Unsharp Mask)"):
    if st.session_state.preprocessed is not None:
        gaussian = cv2.GaussianBlur(st.session_state.preprocessed, (5,5), 0)
        unsharp = cv2.addWeighted(st.session_state.preprocessed, 1.5, gaussian, -0.5, 0)
        st.session_state.enhanced = unsharp
        st.success("Image enhancement complete")
    else:
        st.warning("Perform preprocessing first.")

# Morphological Segmentation
if st.button("Segment Stones"):
    if st.session_state.enhanced is not None:
        binary = cv2.adaptiveThreshold(
            st.session_state.enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        st.session_state.morph = morph
        st.image(morph, caption="Morphological Segmentation", use_column_width=True)
    else:
        st.warning("Enhance image before segmentation.")

# Detect stones by size and shape filtering
if st.button("Detect Kidney Stones"):
    if st.session_state.morph is not None and st.session_state.image is not None:
        labels = measure.label(st.session_state.morph, connectivity=2)
        props = measure.regionprops(labels)

        output = cv2.cvtColor(st.session_state.image, cv2.COLOR_GRAY2BGR)
        stone_found = False
        for region in props:
            if 50 < region.area < 1500:
                circularity = 4 * np.pi * region.area / (region.perimeter ** 2 + 1e-5)
                if circularity > 0.5:
                    y, x = region.centroid
                    minr, minc, maxr, maxc = region.bbox
                    cv2.rectangle(output, (minc, minr), (maxc, maxr), (0,0,255), 2)
                    cv2.circle(output, (int(x), int(y)), 4, (0,255,0), -1)
                    stone_found = True
        if stone_found:
            st.session_state.detected = output
            st.image(output, caption="Detected Kidney Stones", use_column_width=True)
        else:
            st.warning("No kidney stones detected.")
    else:
        st.warning("Perform segmentation first.")
