# Save this as app.py
# Install dependencies:
# pip install streamlit pytesseract easyocr pillow opencv-python

import streamlit as st
from PIL import Image
import pytesseract
import easyocr
import numpy as np
import cv2

# Optional: Set Tesseract OCR path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    # Convert to numpy array and to grayscale
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding for binarization
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def ocr_tesseract(image):
    processed_img = preprocess_image(image)
    text = pytesseract.image_to_string(processed_img, config='--oem 3 --psm 6')
    return text

def ocr_easyocr(image):
    reader = easyocr.Reader(['en'])
    img_array = np.array(image.convert('RGB'))
    result = reader.readtext(img_array, detail=0)
    return "\n".join(result)

def main():
    st.title("Image to Text OCR App with Streamlit")

    ocr_option = st.radio("Choose OCR Engine:", ("Tesseract", "EasyOCR"))

    uploaded_file = st.file_uploader("Upload an image file (jpg, png, jpeg)", type=['jpg','jpeg','png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Extract Text"):
            with st.spinner("Running OCR..."):
                if ocr_option == "Tesseract":
                    extracted_text = ocr_tesseract(image)
                else:
                    extracted_text = ocr_easyocr(image)
            st.success("OCR Complete!")
            st.text_area("Extracted Text", extracted_text, height=300)

if __name__ == "__main__":
    main()
