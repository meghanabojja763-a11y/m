import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

def tumor_detection(img):
    # Process grayscale image using morphological image processing.
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.adaptiveThreshold(closed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            cv2.drawContours(output, [contour], -1, (0, 0, 255), 2)
    return [img, blurred, opened, closed, thresh, clean, output]

def plot_results(images, titles):
    plt.figure(figsize=(15, 8))
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

uploaded_files = st.file_uploader("Upload Medical Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    titles = ['Original', 'Blurred', 'Opening', 'Closing', 'Threshold', 'Cleaned', 'Detected Tumor']
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        st.write(f"Processing {uploaded_file.name}")
        results = tumor_detection(img)
        plot_results(results, titles)
else:
    st.text("Please upload at least one medical image file.")
