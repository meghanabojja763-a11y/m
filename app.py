import streamlit as st
import cv2
import numpy as np
import zipfile
import tempfile
import os
from skimage.metrics import structural_similarity as ssim

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Advanced Image Matching", layout="centered")

st.title("🧠 Advanced Folder Image Matching (Image Processing + ORB + SSIM + DL)")

st.write("""
Upload a **single image** and a **folder (ZIP)** of images.  
This tool will:
- Enhance + morphologically process images  
- Extract visual features using ORB (feature matching)  
- Compare using SSIM (pixel-based)  
- 🔥 **Deep Learning**: CNN embeddings (ResNet-50) + cosine similarity  
- Find whether the image is present in the folder (robust to resize/enhance)
""")

# ------------------- FILE UPLOADS -------------------
query_file = st.file_uploader("📸 Upload the Single Image", type=["jpg", "jpeg", "png"])
folder_zip = st.file_uploader("🗂️ Upload the Folder (as ZIP file containing images)", type=["zip"])

# ------------------- CONTROLS -------------------
st.sidebar.header("Thresholds & Weights")
orb_threshold = st.sidebar.slider("ORB match threshold", 0.0, 1.0, 0.60, 0.01)
ssim_threshold = st.sidebar.slider("SSIM threshold", 0.0, 1.0, 0.70, 0.01)
dl_threshold = st.sidebar.slider("DL cosine threshold", 0.0, 1.0, 0.85, 0.01)

w_dl = st.sidebar.slider("Weight: DL", 0.0, 1.0, 0.50, 0.05)
w_orb = st.sidebar.slider("Weight: ORB", 0.0, 1.0, 0.30, 0.05)
w_ssim = st.sidebar.slider("Weight: SSIM", 0.0, 1.0, 0.20, 0.05)

norm = max(1e-9, (w_dl + w_orb + w_ssim))
w_dl, w_orb, w_ssim = w_dl/norm, w_orb/norm, w_ssim/norm

# ------------------- HELPER FUNCTIONS -------------------
def read_color_for_dl(image_bytes):
    """Read color image (BGR) for DL embedding (no equalize/morph)."""
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")
    return img

def preprocess_image(image_bytes):
    """Enhance, morphologically filter, and prepare an image for ORB/SSIM (grayscale)."""
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")

    # Resize for uniform comparison
    img = cv2.resize(img, (400, 400))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Enhancement: Histogram Equalization ---
    enhanced = cv2.equalizeHist(gray)

    # --- Morphological Operations ---
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    return morph

def orb_similarity(img1, img2):
    """Compute ORB feature matching score."""
    orb = cv2.ORB_create(1000)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) == 0:
        return 0.0

    distances = [m.distance for m in matches]
    avg_distance = sum(distances) / len(distances)

    # Normalize score: smaller distance = better match
    similarity = max(0.0, min(1.0, 1.0 - avg_distance / 100.0))
    return float(similarity)

def ssim_similarity(img1, img2):
    """Compute SSIM similarity for verification."""
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    score, _ = ssim(img1, img2, full=True)
    return float(score)

# ---- DL: ResNet50 embedding & cosine similarity ----
_DL_OK = True
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torchvision.models import resnet50, ResNet50_Weights
except Exception as e:
    _DL_OK = False
    _DL_IMPORT_ERR = str(e)

@st.cache_resource(show_spinner=False)
def load_resnet50_backbone():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    # Remove final classification layer -> global avg pooled 2048-dim embedding
    backbone = nn.Sequential(*list(model.children())[:-1])
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    preprocess = weights.transforms()
    return backbone, preprocess

def img_to_embedding_bgr(img_bgr, backbone, preprocess, device):
    """
    img_bgr: OpenCV BGR image (H, W, 3)
    Returns L2-normalized embedding vector (2048,)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    import PIL.Image as Image
    pil_img = Image.fromarray(img_rgb)
    x = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = backbone(x)
    feat = feat.view(-1)
    feat = feat / (torch.norm(feat) + 1e-9)
    return feat.cpu().numpy()

def dl_cosine_similarity(img_bgr_a, img_bgr_b, backbone, preprocess, device):
    try:
        va = img_to_embedding_bgr(img_bgr_a, backbone, preprocess, device)
        vb = img_to_embedding_bgr(img_bgr_b, backbone, preprocess, device)
        cos = float(np.dot(va, vb))
        return (cos + 1.0) / 2.0
    except Exception:
        return 0.0

def extract_all_images(zip_path, extract_to):
    """Extract all images (including nested folders) from ZIP."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    image_paths = []
    for root, _, files in os.walk(extract_to):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, f))
    return image_paths

# ------------------- MAIN LOGIC -------------------
if query_file and folder_zip:
    with st.spinner("Processing images..."):
        query_bytes = query_file.read()
        query_img_proc = preprocess_image(query_bytes)

        temp_dir = tempfile.mkdtemp()
        image_paths = extract_all_images(folder_zip, temp_dir)
        st.info(f"Found {len(image_paths)} images in the folder.")

        use_dl = True
        if not _DL_OK:
            use_dl = False
            st.warning(
                "Deep Learning check disabled (PyTorch/torchvision not available).\n\n"
                f"Import error: `{_DL_IMPORT_ERR}`\n\n"
                "Install with:\n`pip install torch torchvision --upgrade`"
            )

        if use_dl:
            try:
                backbone, preprocess_t = load_resnet50_backbone()
                device = "cuda" if False and torch.cuda.is_available() else "cpu"
                if device == "cuda":
                    backbone.to(device)
                query_img_color_for_dl = read_color_for_dl(query_bytes)
            except Exception as e:
                use_dl = False
                st.warning(f"Deep Learning check disabled at runtime: {e}")

        matches = []
        all_results = []

        prog = st.progress(0.0)
        n = max(1, len(image_paths))

        for i, path in enumerate(image_paths):
            with open(path, "rb") as f:
                folder_bytes = f.read()

            folder_img_proc = preprocess_image(folder_bytes)
            orb_score = orb_similarity(query_img_proc, folder_img_proc)
            ssim_score = ssim_similarity(query_img_proc, folder_img_proc)

            dl_score = 0.0
            if use_dl:
                try:
                    folder_img_color_for_dl = read_color_for_dl(folder_bytes)
                    dl_score = dl_cosine_similarity(
                        query_img_color_for_dl, folder_img_color_for_dl, backbone, preprocess_t, device
                    )
                except Exception:
                    dl_score = 0.0

            combined = (w_dl * dl_score) + (w_orb * orb_score) + (w_ssim * ssim_score)
            all_results.append((path, orb_score, ssim_score, dl_score, combined))

            if (orb_score >= orb_threshold) or (ssim_score >= ssim_threshold) or (dl_score >= dl_threshold):
                matches.append((path, orb_score, ssim_score, dl_score, combined))

            prog.progress((i + 1) / n)

        # --- Display Results ---
        if matches:
            st.success(f"✅ Found {len(matches)} possible match(es)!")
            for path, orb_s, ssim_s, dl_s, comb in sorted(matches, key=lambda x: x[4], reverse=True):
                st.write(
                    f"**{os.path.basename(path)}** — "
                    f"DL: `{dl_s:.3f}`, ORB: `{orb_s:.3f}`, SSIM: `{ssim_s:.3f}`, "
                    f"**Combined:** `{comb:.3f}`"
                )
                st.image(path, caption=f"Matched Image (Combined: {comb:.3f})", use_container_width=True)
        else:
            st.error("❌ No matching image found.")

else:
    st.info("Please upload a single image and a folder (as ZIP).")
