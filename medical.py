import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def tumor_detection(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
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
            cv2.drawContours(output, [contour], -1, (0,0,255), 2)

    plt.figure(figsize=(15,8))
    titles = ['Original', 'Blurred', 'Opening', 'Closing', 'Threshold', 'Cleaned', 'Detected Tumor']
    images = [img, blurred, opened, closed, thresh, clean, output]
    for i in range(len(images)):
        plt.subplot(2,4,i+1)
        if i < len(images)-1:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Get folder path from user input (or specify static path)
dataset_folder = input("Enter path to dataset folder: ")

# Process all PNG/JPG images in given folder
for file in os.listdir(dataset_folder):
    if file.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(dataset_folder, file)
        print(f"Processing {file} ...")
        tumor_detection(image_path)
