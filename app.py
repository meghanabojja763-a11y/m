import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
import io

st.title("🖼️ Image Compression App")
st.write("Upload an image and compress it to a target size (in KB or MB).")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)

    # Get target size
    col1, col2 = st.columns(2)
    with col1:
        target_value = st.number_input("Enter target size", min_value=10.0, value=200.0)
    with col2:
        target_unit = st.selectbox("Select unit", ["KB", "MB"])

    if st.button("Compress Image"):
        # Convert MB → KB if needed
        if target_unit == "MB":
            target_size_kb = target_value * 1024
        else:
            target_size_kb = target_value

        # Convert image to OpenCV format
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Save temporarily
        temp_path = "temp_input.jpg"
        cv2.imwrite(temp_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Open using PIL for compression
        img = Image.open(temp_path)
        output_path = "compressed_output.jpg"
        quality = 95
        target_size = target_size_kb * 1024  # KB → Bytes

        while True:
            img.save(output_path, "JPEG", quality=quality)
            size = os.path.getsize(output_path)
            if size <= target_size or quality <= 10:
                break
            quality -= 5

        compressed_img = Image.open(output_path)
        compressed_size_kb = size / 1024

        # Display results
        colA, colB = st.columns(2)
        with colA:
            st.image(image, caption=f"Original ({uploaded_file.size/1024:.2f} KB)", use_container_width=True)
        with colB:
            st.image(compressed_img, caption=f"Compressed ({compressed_size_kb:.2f} KB)", use_container_width=True)

        # Provide download option
        with open(output_path, "rb") as f:
            btn = st.download_button(
                label="📥 Download Compressed Image",
                data=f,
                file_name="compressed.jpg",
                mime="image/jpeg"
            )

        os.remove(temp_path)
