import streamlit as st
from PIL import Image
import io
import os

st.title("🖼️ Image Compression App")
st.write("Upload an image and compress it to a target size (in KB or MB).")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Display original image
    original_bytes = uploaded_file.getbuffer()
    original_size_kb = len(original_bytes) / 1024
    st.image(image, caption=f"Original Image ({original_size_kb:.2f} KB)", use_container_width=True)

    # Input fields
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

        # Initialize compression variables
        quality = 95
        step = 5
        output_buffer = io.BytesIO()

        # Iteratively reduce quality until target size is reached
        while quality > 5:
            output_buffer = io.BytesIO()
            image.save(output_buffer, format="JPEG", quality=quality)
            size_kb = len(output_buffer.getvalue()) / 1024
            if size_kb <= target_size_kb:
                break
            quality -= step

        compressed_size_kb = size_kb
        compression_ratio = (1 - (compressed_size_kb / original_size_kb)) * 100

        # Display results
        colA, colB = st.columns(2)
        with colA:
            st.image(image, caption=f"Original ({original_size_kb:.2f} KB)", use_container_width=True)
        with colB:
            st.image(output_buffer.getvalue(), caption=f"Compressed ({compressed_size_kb:.2f} KB)", use_container_width=True)

        st.success(f"✅ Compression complete! Reduced by {compression_ratio:.2f}%")

        # Download button
        st.download_button(
            label="📥 Download Compressed Image",
            data=output_buffer.getvalue(),
            file_name="compressed.jpg",
            mime="image/jpeg"
        )
