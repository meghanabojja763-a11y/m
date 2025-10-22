import streamlit as st
from PIL import Image
import io, zipfile
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

st.title("🖼️ Smart Image Compression & Expansion App")
st.write("""
Upload multiple images, set a target size (KB/MB), preserve faces/text, 
choose output format (JPEG/WebP), and download all processed images in a ZIP.
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

            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # --- Face detection ---
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
            for (x, y, w, h) in faces:
                mask[y:y+h, x:x+w] = 255

            output_buffer = io.BytesIO()

            # --- Compression Mode ---
            if target_size_kb < original_size_kb:
                quality = 95
                step = 5
                while quality > 5:
                    temp_buffer = io.BytesIO()
                    image.save(temp_buffer, format=output_format, quality=quality)
                    size_kb = len(temp_buffer.getvalue()) / 1024
                    if size_kb <= target_size_kb:
                        output_buffer = temp_buffer
                        break
                    quality -= step
                else:
                    image.save(output_buffer, format=output_format, quality=quality)

            # --- Expansion Mode ---
            else:
                scale_factor = 1.0
                max_iterations = 15
                for _ in range(max_iterations):
                    # Resize image
                    new_w = int(image.width * scale_factor)
                    new_h = int(image.height * scale_factor)
                    resized_img = image.resize((new_w, new_h), Image.LANCZOS)

                    temp_buffer = io.BytesIO()
                    resized_img.save(temp_buffer, format=output_format, quality=100)
                    size_kb = len(temp_buffer.getvalue()) / 1024

                    if size_kb >= target_size_kb * 0.95:  # close to target
                        output_buffer = temp_buffer
                        break

                    # Adjust scale based on difference ratio
                    scale_factor *= (target_size_kb / max(size_kb, 1)) ** 0.3

                else:
                    # fallback to largest generated image
                    output_buffer = temp_buffer

            # --- Quality Metrics ---
            output_buffer.seek(0)
            processed_image = Image.open(output_buffer)
            processed_np = np.array(processed_image)
            original_np = np.array(image)

            if processed_np.shape != original_np.shape:
                processed_np = cv2.resize(processed_np, (original_np.shape[1], original_np.shape[0]))

            mse = np.mean((original_np - processed_np) ** 2)
            psnr_val = 10 * np.log10(255**2 / mse) if mse != 0 else 100
            try:
                ssim_val = ssim(original_np, processed_np, channel_axis=-1)
            except ValueError:
                ssim_val = 0.0

            # --- Display ---
            st.write(f"**{uploaded_file.name}**")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption=f"Original ({original_size_kb:.2f} KB)")
            with col2:
                size_kb = len(output_buffer.getvalue()) / 1024
                st.image(processed_image, caption=f"Processed ({size_kb:.2f} KB)")

            st.write(f"📊 PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")

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
