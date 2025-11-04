import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor, as_completed

def preprocess_image(image_path, size=(128, 128)):
    """Read and preprocess image: enhance, resize, and apply morphology."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    resized = cv2.resize(denoised, size, interpolation=cv2.INTER_AREA)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel)
    return morph


def quick_compare(img1, img2):
    """Fast comparison using SSIM and histogram only."""
    ssim_score, _ = ssim(img1, img2, full=False)
    hist1 = cv2.calcHist([img1], [0], None, [64], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [64], [0, 256])
    hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return 0.7 * ssim_score + 0.3 * hist_score


def detailed_compare(img1, img2):
    """More accurate but slower comparison including ORB features."""
    ssim_score, _ = ssim(img1, img2, full=True)
    hist1 = cv2.calcHist([img1], [0], None, [64], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [64], [0, 256])
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
    return 0.5 * ssim_score + 0.3 * hist_score + 0.2 * match_score


def search_image_in_dataset(dataset_folder, search_image_path, threshold=0.85):
    """Search dataset for image similar to search image, optimized version."""
    print(f"\n🔍 Searching for image similar to: {os.path.basename(search_image_path)}")
    search_img = preprocess_image(search_image_path)

    # --- Step 1: Cache preprocessed dataset images ---
    dataset_images = []
    for filename in os.listdir(dataset_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            dataset_images.append((filename, preprocess_image(os.path.join(dataset_folder, filename))))

    # --- Step 2: Quick parallel comparison ---
    quick_scores = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(quick_compare, search_img, img): fname for fname, img in dataset_images}
        for future in as_completed(futures):
            filename = futures[future]
            try:
                score = future.result()
                quick_scores.append((filename, score))
            except Exception as e:
                print(f"Error on {filename}: {e}")

    # --- Step 3: Select top candidates for detailed comparison ---
    quick_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = quick_scores[:min(5, len(quick_scores))]

    print("\nTop quick matches (before ORB):")
    for f, s in top_candidates:
        print(f"{f:30} -> Score: {s:.4f}")

    # --- Step 4: Detailed check for top matches ---
    best_match = None
    best_score = 0
    for filename, _ in top_candidates:
        dataset_img = [img for f, img in dataset_images if f == filename][0]
        score = detailed_compare(search_img, dataset_img)
        print(f"Detailed score for {filename}: {score:.4f}")
        if score > best_score:
            best_score = score
            best_match = filename

    # --- Step 5: Decision ---
    print("\n=== Search Results ===")
    if best_score >= threshold:
        print(f"✅ Match Found: {best_match} (Score: {best_score:.4f})")
        return {
            "filename": best_match,
            "path": os.path.join(dataset_folder, best_match),
            "similarity_score": best_score
        }
    else:
        print("❌ No similar image found in dataset.")
        return None


if __name__ == "__main__":
    dataset_folder = input("Enter dataset folder path: ").strip()
    search_image_path = input("Enter search image path: ").strip()

    result = search_image_in_dataset(dataset_folder, search_image_path)
    if result:
        print("\nImage Information:")
        for k, v in result.items():
            print(f"{k}: {v}")
