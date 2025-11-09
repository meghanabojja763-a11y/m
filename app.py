import streamlit as st
import numpy as np
import zipfile
import tempfile
import os

# Try importing OpenCV
try:
    import cv2
    _CV2_OK = True
except Exception as e:
    _CV2_OK = False
    _CV2_ERR = str(e)

# Try importing SSIM
try:
    from skimage.metrics import structural_similarity as ssim
    _SSIM_OK = True
except Exception as e:
    _SSIM_OK = False
    _SSIM_ERR = str(e)

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
def safe_read_color(image_bytes):
    if not _CV2_OK:
        raise RuntimeError("OpenCV not available.")
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")
    return img

def preprocess_image(image_bytes):
    if not _CV2_OK:
        raise RuntimeError("OpenCV not available.")
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")

    img = cv2.resize(img, (400, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    return morph

def orb_similarity(img1, img2):
    if not _CV2_OK:
        return 0.0
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
    return max(0.0, min(1.0, 1.0 - avg_distance / 100.0))

def ssim_similarity(img1, img2):
    if not _SSIM_OK:
        return 0.0
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    score, _ = ssim(img1, img2, full=True)
    return float(score)

# ---- DL (ResNet50) ----
_DL_OK = True
try:
    import torch
    import torch.nn as nn
    from torchvision.models import resnet50, ResNet50_Weights
except Exception as e:
    _DL_OK = False
    _DL_IMPORT_ERR = str(e)

@st.cache_resource(show_spinner=False)
def load_resnet50_backbone():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    backbone = nn.Sequential(*list(model.children())[:-1])
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    preprocess = weights.transforms()
    return backbone, preprocess

def img_to_embedding_bgr(img_bgr, backbone, preprocess, device):
    import cv2
    import PIL.Image as Image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    x = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = backbone(x)
    feat = feat.view(-1)
    feat = feat / (torch.norm(feat) + 1e-9)
    return feat.cpu().numpy()

def dl_cosine_similarity(img_a, img_b, backbone, preprocess, device):
    try:
        va = img_to_embedding_bgr(img_a, backbone, preprocess, device)
        vb = img_to_embedding_bgr(img_b, backbone, preprocess, device)
        cos = float(np.dot(va, vb))
        return (cos + 1.0) / 2.0
    except Exception:
        return 0.0

def extract_all_images(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    image_paths = []
    for root, _, files in os.walk(extract_to):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, f))
    return image_paths

# ------------------- MAIN LOGIC -------------------
if not _CV2_OK:
    st.error(f"❌ OpenCV not available: {_CV2_ERR}\n\nInstall it using:\n`pip install opencv-python-headless`")
elif query_file and folder_zip:
    with st.spinner("Processing images..."):
        query_bytes = query_file.read()
        query_img_proc = preprocess_image(query_bytes)
        temp_dir = tempfile.mkdtemp()
        image_paths = extract_all_images(folder_zip, temp_dir)
        st.info(f"Found {len(image_paths)} images in the folder.")

        use_dl = _DL_OK
        if not _DL_OK:
            st.warning(
                "Deep Learning check disabled (PyTorch/torchvision not available).\n\n"
                f"Import error: `{_DL_IMPORT_ERR}`\n\n"
                "Install with:\n`pip install torch torchvision --upgrade`"
            )

        if use_dl:
            try:
                backbone, preprocess_t = load_resnet50_backbone()
                device = "cpu"
                query_img_color_for_dl = safe_read_color(query_bytes)
            except Exception as e:
                use_dl = False
                st.warning(f"Deep Learning check disabled at runtime: {e}")

        matches = []
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
                    folder_img_color = safe_read_color(folder_bytes)
                    dl_score = dl_cosine_similarity(
                        query_img_color_for_dl, folder_img_color, backbone, preprocess_t, device
                    )
                except Exception:
                    dl_score = 0.0

            combined = (w_dl * dl_score) + (w_orb * orb_score) + (w_ssim * ssim_score)
            if (orb_score >= orb_threshold) or (ssim_score >= ssim_threshold) or (dl_score >= dl_threshold):
                matches.append((path, orb_score, ssim_score, dl_score, combined))

            prog.progress((i + 1) / n)

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
