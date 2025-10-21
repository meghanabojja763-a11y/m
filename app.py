import streamlit as st
import cv2
import numpy as np
from skimage import measure

st.set_page_config(page_title="Kidney Stone Detection", layout="wide")
st.title("Kidney Stone Detection from X-ray Images")

# Initialize session_state
for key in ["image", "processed", "enhanced", "filtered", "morph", "result"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Step 1: Upload Image
uploaded_file = st.file_uploader("Browse and upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    st.session_state.image = cv2.resize(image, (256, 256))
    st.image(st.session_state.image, caption="Original X-ray Image", use_column_width=True)

# --- STEP 1: Preprocessing ---
if st.button("Run Preprocessing"):
    if st.session_state.image is not None:
        img = st.session_state.image
        blurred = cv2.medianBlur(img, 5)
        equalized = cv2.equalizeHist(blurred)
        st.session_state.processed = equalized
        st.image(equalized, caption="Preprocessed Image (After Step 1)", use_column_width=True)
    else:
        st.warning("Please upload an image first.")

# --- STEP 2: Enhancement (CLAHE + Unsharp Mask) ---
if st.button("Enhance Image"):
    if st.session_state.processed is not None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(st.session_state.processed)
        gaussian = cv2.GaussianBlur(clahe_img, (3, 3), 0)
        unsharp = cv2.addWeighted(clahe_img, 1.5, gaussian, -0.5, 0)
        st.session_state.enhanced = unsharp
        st.success("Enhancement applied successfully.")
    else:
        st.warning("Please run preprocessing before enhancement.")

# --- STEP 3: Filtering ---
if st.button("Apply Filtering"):
    if st.session_state.enhanced is not None:
        filtered = cv2.medianBlur(st.session_state.enhanced, 3)
        st.session_state.filtered = filtered
        st.success("Filtering completed successfully.")
    else:
        st.warning("Please enhance the image before filtering.")

# --- STEP 4: Morphological Processing ---
if st.button("Morphological Processing"):
    if st.session_state.filtered is not None:
        _, binary = cv2.threshold(st.session_state.filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        st.session_state.morph = morph
        st.success("Morphological segmentation complete.")
    else:
        st.warning("Please apply filtering before morphological processing.")

# --- STEP 5: Detection (Centroid & Area) ---
if st.button("Detect Kidney Stone (Centroid and Area)"):
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
            st.image(output, caption="Detected Kidney Stone", use_column_width=True)
            st.success(f"Kidney stone detected. Centroid: ({int(centroid[1])}, {int(centroid[0])}) | Area: {int(area)} pixels")
        else:
            st.warning("No kidney stone detected in the image.")
    else:
        st.warning("Please perform morphological processing first.")
