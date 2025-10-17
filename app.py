import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import tempfile
import os

def tumor_detection(img):
    # Morphological tumor detection pipeline on grayscale image
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.adaptiveThreshold(
        closed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            cv2.drawContours(output, [contour], -1, (0, 0, 255), 2)
    return [img, blurred, opened, closed, thresh, clean, output]

def plot_results(images, titles):
    plt.figure(figsize=(16, 8))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 4, i + 1)
        if i < len(images) - 1:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    st.pyplot(plt)
    plt.clf()

st.title("Medical Tumor Detection with Morphological Image Processing")

uploaded_zip = st.file_uploader(
    "Upload a ZIP file containing medical images (PNG, JPG, JPEG)",
    type=["zip"]
)

if uploaded_zip is not None:
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())

        # Extract all files
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find image files inside extracted directory
        image_extensions = (".png", ".jpg", ".jpeg")
        image_files = [os.path.join(temp_dir, file)
                       for file in os.listdir(temp_dir)
                       if file.lower().endswith(image_extensions)]

        if not image_files:
            st.warning("No valid image files found in ZIP.")
        else:
            titles = ['Original', 'Blurred', 'Opening', 'Closing', 'Threshold', 'Cleaned', 'Detected Tumor']
            for image_file in image_files:
                st.write(f"Processing: {os.path.basename(image_file)}")
                img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    results = tumor_detection(img)
                    plot_results(results, titles)
                else:
                    st.error(f"Could not read {os.path.basename(image_file)}")
else:
    st.text("Please upload a ZIP file containing one or more medical images to analyze.")
