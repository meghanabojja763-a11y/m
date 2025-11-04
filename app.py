import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def preprocess_image(image_path):
    """Read and preprocess image: enhance, compress, and apply morphology."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    resized = cv2.resize(denoised, (256, 256), interpolation=cv2.INTER_AREA)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel)
    return morph


def compare_images(img1, img2):
    """Return combined similarity score between two preprocessed images."""
    ssim_score, _ = ssim(img1, img2, full=True)
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        match_score = len(matches) / max(len(kp1), len(kp2))
    else:
        match_score = 0

    # Weighted combination
    combined_score = (0.5 * ssim_score) + (0.3 * hist_score) + (0.2 * match_score)
    return combined_score, (ssim_score, hist_score, match_score)


def search_image_in_dataset(dataset_folder, search_image_path, threshold=0.85):
    """Search the dataset for an image similar to the given search image."""
    search_img = preprocess_image(search_image_path)
    print(f"\n🔍 Searching for image similar to: {os.path.basename(search_image_path)}")
    
    best_match = None
    best_score = 0
    all_scores = []

    for filename in os.listdir(dataset_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            dataset_img_path = os.path.join(dataset_folder, filename)
            try:
                dataset_img = preprocess_image(dataset_img_path)
                score, (s, h, m) = compare_images(search_img, dataset_img)
                all_scores.append((filename, score))
                print(f"{filename:30}  -> Combined Score: {score:.4f}  (SSIM={s:.3f}, HIST={h:.3f}, MATCH={m:.3f})")

                if score > best_score:
                    best_score = score
                    best_match = filename
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print("\n=== Search Results ===")
    if best_score >= threshold:
        print(f"✅ Match Found: {best_match}  (Score: {best_score:.4f})")
        # Suppose info is stored as a dictionary (you can modify as needed)
        image_info = {
            "filename": best_match,
            "path": os.path.join(dataset_folder, best_match),
            "similarity_score": best_score
        }
        return image_info
    else:
        print("❌ No similar image found in dataset.")
        return None


# Example usage
if __name__ == "__main__":
    dataset_folder = input("Enter dataset folder path: ").strip()
    search_image_path = input("Enter search image path: ").strip()
    
    result = search_image_in_dataset(dataset_folder, search_image_path)

    if result:
        print("\nImage Information:")
        for key, value in result.items():
            print(f"{key}: {value}")
