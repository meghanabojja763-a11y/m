import streamlit as st
from PIL import Image
import io

st.title("🖼️ Image Compression & Expansion App")
st.write("Upload an image and set a target size (in KB or MB).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    original_bytes = uploaded_file.getbuffer()
    original_size_kb = len(original_bytes) / 1024
    st.image(image, caption=f"Original Image ({original_size_kb:.2f} KB)", use_container_width=True)

    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        target_value = st.number_input("Enter target size", min_value=1.0, value=200.0)
    with col2:
        target_unit = st.selectbox("Select unit", ["KB", "MB"])

    if st.button("Process Image"):
        # Convert MB → KB
        target_size_kb = target_value * 1024 if target_unit == "MB" else target_value

        # Initialize buffer
        output_buffer = io.BytesIO()

        # COMPRESSION
        if target_size_kb < original_size_kb:
            quality = 95
            step = 5
            while quality > 5:
                output_buffer = io.BytesIO()
                image.save(output_buffer, format="JPEG", quality=quality)
                size_kb = len(output_buffer.getvalue()) / 1024
                if size_kb <= target_size_kb:
                    break
                quality -= step

        # EXPANSION
        else:
            # Save at max quality first
            output_buffer = io.BytesIO()
            image.save(output_buffer, format="JPEG", quality=100)
            size_kb = len(output_buffer.getvalue()) / 1024

            # If still smaller than target, resize slightly
            if size_kb < target_size_kb:
                scale_factor = (target_size_kb / size_kb) ** 0.5
                new_w = max(1, int(image.width * scale_factor))
                new_h = max(1, int(image.height * scale_factor))
                enlarged = image.resize((new_w, new_h), Image.LANCZOS)
                output_buffer = io.BytesIO()
                enlarged.save(output_buffer, format="JPEG", quality=100)
                size_kb = len(output_buffer.getvalue()) / 1024

        # Convert KB → MB if needed
        display_size = size_kb / 1024 if target_unit == "MB" else size_kb
        display_unit = "MB" if target_unit == "MB" else "KB"

        # Percentage change
        change = ((size_kb - original_size_kb) / original_size_kb) * 100
        direction = "increased" if change > 0 else "reduced"

        # Display result safely using PIL.Image.open
        processed_image = Image.open(output_buffer)
        colA, colB = st.columns(2)
        with colA:
            st.image(image, caption=f"Original ({original_size_kb:.2f} KB)", use_container_width=True)
        with colB:
            st.image(processed_image, caption=f"Processed ({display_size:.2f} {display_unit})", use_container_width=True)

        st.success(f"✅ Image {direction} by {abs(change):.2f}%. Final size: {display_size:.2f} {display_unit}")

        # Download button
        st.download_button(
            label="📥 Download Processed Image",
            data=output_buffer.getvalue(),
            file_name="processed_image.jpg",
            mime="image/jpeg"
        )
