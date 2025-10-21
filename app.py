import streamlit as st
import cv2
import numpy as np
from skimage import measure

st.set_page_config(page_title="Kidney Stone Detection", layout="wide")
st.title("Kidney Stone Detection from X-ray Images")

# Initialize session_state
if "image" not in st.session_state:
    st.session_state.image = None
if "processed" not in st.session_state:
    st.session_state.processed = None
if "enhanced" not in st.session_state:
    st.session_state.enhanced = None
if "filtered" not in st.session_state:
    st.session_state.filtered = None
if "morph" not in st.session_state:
    st.session_state.morph = None
if "result" not in st.session_state:
    st.session_state.result = None

uploaded_file = st.file_uploader("Browse and upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    st.session_state.image = cv2.resize(image, (256, 256))
    st.image(st.session_state.image, caption="Original Image", use_column_width=True)

# --- STEP 1: Preprocessing ---
if st.button("Preprocessing (Resize, Median Filter, Histogram Equalization)"):
    if st.session_state.image is not None:
        img = st.session_state.image
        blurred = cv2.medianBlur(img, 5)
        equalized = cv2.equalizeHist(blurred)
        st.session_state.processed = equalized
        st.image(equalized, caption="Preprocessed Image", use_column_width=True)
    else:
        st.warning("Please upload an image first.")

# --- STEP 2: Image Enhancement ---
if st.button("Enhance Image (CLAHE + Unsharp Mask)"):
    if st.session_state.processed is not None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(st.session_state.processed)
        gaussian = cv2.GaussianBlur(clahe_img, (3,3), 0)
        unsharp = cv2.addWeighted(clahe_img, 1.5, gaussian, -0.5, 0)
        st.session_state.enhanced = unsharp
        st.image(unsharp, caption="Enhanced Image", use_column_width=True)
    else:
        st.warning("Perform preprocessing first.")

# --- STEP 3: Filtering ---
if st.button("Apply Filter (Median + Sharpness Boost)"):
    if st.session_state.enhanced is not None:
        filtered = cv2.medianBlur(st.session_state.enhanced, 3)
        st.session_state.filtered = filtered
        st.image(filtered, caption="Filtered Image", use_column_width=True)
    else:
        st.warning("Enhance image before applying filter.")

# --- STEP 4: Morphological Processing ---
if st.button("Morphological Segmentation"):
    if st.session_state.filtered is not None:
        _, thresh = cv2.threshold(st.session_state.filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        st.session_state.morph = morph
        st.image(morph, caption="Morphological Segmentation Result", use_column_width=True)
    else:
        st.warning("Apply filtering first.")

# --- STEP 5: Centroid and Area of Stone ---
if st.button("Calculate Centroid and Area"):
    if st.session_state.morph is not None:
        labels = measure.label(st.session_state.morph, connectivity=2)
        props = measure.regionprops(labels)

        output = cv2.cvtColor(st.session_state.image, cv2.COLOR_GRAY2BGR)
        if props:
            largest = max(props, key=lambda x: x.area)
            area = largest.area
            centroid = largest.centroid
            minr, minc, maxr, maxc = largest.bbox
            cv2.rectangle(output, (minc, minr), (maxc, maxr), (0, 0, 255), 2)
            cv2.circle(output, (int(centroid[1]), int(centroid[0])), 5, (0, 255, 0), -1)
            st.image(output, caption="Detected Stone", use_column_width=True)
            st.success(f"Centroid: ({int(centroid[1])}, {int(centroid[0])}) | Area: {int(area)} pixels")
        else:
            st.warning("No kidney stone region detected.")
    else:
        st.warning("Run morphological segmentation first.")
