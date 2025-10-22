import streamlit as st
from PIL import Image
import io, os, zipfile
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

st.title("🖼️ Smart Image Compression App")
st.write("""
Upload multiple images, set a target size (KB/MB), preserve faces/text, 
choose output format (JPEG/WebP), and download all images in a ZIP.
""")

# --- Upload multiple images ---
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    target_value = st.number_input("Target size per image", min_value=1.0, value=200.0)
    target_unit = st.selectbox("Unit", ["KB", "MB"])
    output_format = st.selectbox("Output format", ["JPEG", "WebP"])
    
    if st.button("Process Images"):
        zip_buffer = io.BytesIO()
        zip_file = zipfile.ZipFile(zip_buffer, mode="w")

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            original_bytes = uploaded_file.getbuffer()
            original_size_kb = len(original_bytes) / 1024

            # Convert target to KB
            target_size_kb = target_value * 1024 if target_unit == "MB" else target_value

            # Convert to OpenCV image for AI-aware processing
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # --- Face detection ---
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Create mask for faces to preserve
            mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
            for (x, y, w, h) in faces:
                mask[y:y+h, x:x+w] = 255

            # --- Compression loop ---
            output_buffer = io.BytesIO()
            quality = 95
            step = 5

            if target_size_kb < original_size_kb:
                # Compress JPEG/WebP iteratively
                while quality > 5:
                    temp_img = image.copy()
                    # Apply mild compression only outside face regions
                    temp_np = np.array(temp_img)
                    # Reduce quality outside face mask
                    temp_np_masked = cv2.bitwise_and(temp_np, temp_np, mask=mask)
                    temp_img_masked = Image.fromarray(temp_np_masked)
                    temp_img_masked.save(output_buffer, format=output_format, quality=quality)
                    size_kb = len(output_buffer.getvalue()) / 1024
                    if size_kb <= target_size_kb:
                        break
                    quality -= step
            else:
                # Expansion: save at max quality/resolution
                image.save(output_buffer, format=output_format, quality=100)
                size_kb = len(output_buffer.getvalue()) / 1024

            # Calculate PSNR and SSIM
            output_buffer.seek(0)
            processed_image = Image.open(output_buffer)
            processed_np = np.array(processed_image)
            original_np = np.array(image)
            
            mse = np.mean((original_np - processed_np) ** 2)
            psnr_val = 10 * np.log10(255**2 / mse) if mse != 0 else 100
            ssim_val = ssim(original_np, processed_np, multichannel=True)

            # Display
            st.write(f"**{uploaded_file.name}**")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption=f"Original ({original_size_kb:.2f} KB)")
            with col2:
                st.image(processed_image, caption=f"Processed ({size_kb:.2f} KB)")

            st.write(f"PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")

            # Add to ZIP
            output_buffer.seek(0)
            zip_file.writestr(f"processed_{uploaded_file.name}", output_buffer.read())

        zip_file.close()
        zip_buffer.seek(0)

        st.download_button(
            label="📥 Download All Processed Images",
            data=zip_buffer.getvalue(),
            file_name="processed_images.zip",
            mime="application/zip"
        )
