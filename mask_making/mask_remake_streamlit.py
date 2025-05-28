import os
import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import maxflow
from PIL import Image

# --- CONFIG ---
IMAGE_FOLDER = "data/To_Remask_Images_Ash"
MASK_OUTPUT_FOLDER = "data/new_masks/masks"
CUTOUT_OUTPUT_FOLDER = "data/new_masks/cutouts"
os.makedirs(MASK_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CUTOUT_OUTPUT_FOLDER, exist_ok=True)

# --- UTILS ---
def graphcut(image, scribble_mask, lambda_val=50, sigma=10):
    H, W, _ = image.shape
    img = image.astype(np.float64)
    fg_pixels = scribble_mask == 1
    bg_pixels = scribble_mask == 2
    fg_mean = np.mean(img[fg_pixels], axis=0) if np.any(fg_pixels) else np.zeros(3)
    bg_mean = np.mean(img[bg_pixels], axis=0) if np.any(bg_pixels) else np.zeros(3)
    data_cost_fg = np.sum((img - fg_mean) ** 2, axis=2)
    data_cost_bg = np.sum((img - bg_mean) ** 2, axis=2)
    data_cost_fg = data_cost_fg / (np.max(data_cost_fg) + 1e-8)
    data_cost_bg = data_cost_bg / (np.max(data_cost_bg) + 1e-8)
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((H, W))
    source_cap = data_cost_fg.copy()
    sink_cap = data_cost_bg.copy()
    source_cap[scribble_mask == 1] = 0
    sink_cap[scribble_mask == 1] = 1e9
    source_cap[scribble_mask == 2] = 1e9
    sink_cap[scribble_mask == 2] = 0
    g.add_grid_tedges(nodeids, source_cap, sink_cap)
    for i in range(H):
        for j in range(W):
            if j < W - 1:
                diff = np.linalg.norm(img[i, j] - img[i, j + 1])
                weight = lambda_val * np.exp(- (diff ** 2) / (2 * sigma ** 2))
                g.add_edge(nodeids[i, j], nodeids[i, j + 1], weight, weight)
            if i < H - 1:
                diff = np.linalg.norm(img[i, j] - img[i + 1, j])
                weight = lambda_val * np.exp(- (diff ** 2) / (2 * sigma ** 2))
                g.add_edge(nodeids[i, j], nodeids[i + 1, j], weight, weight)
    g.maxflow()
    seg_mask = np.zeros((H, W), dtype=bool)
    for i in range(H):
        for j in range(W):
            seg_mask[i, j] = True if g.get_segment(nodeids[i, j]) == 0 else False
    return seg_mask

def check_if_done(filename_base):
    mask_path = os.path.join(MASK_OUTPUT_FOLDER, f"{filename_base}_mask.png")
    cutout_path = os.path.join(CUTOUT_OUTPUT_FOLDER, f"{filename_base}_chopped.png")
    return os.path.exists(mask_path) and os.path.exists(cutout_path)

# --- MAIN APP ---
st.set_page_config(page_title="Mask Maker", layout="wide")
st.title("Mask Making Tool (Graph-Cut)")

# Load image list
image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
if not image_files:
    st.error(f"No images found in {IMAGE_FOLDER}. Please add .png, .jpg, or .jpeg files.")
    st.stop()
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

# Navigation
col_nav, col_status = st.columns([3, 1])
with col_nav:
    prev = st.button("← Previous Image")
    next = st.button("→ Next Image")
with col_status:
    filename_base = os.path.splitext(image_files[st.session_state.current_index])[0]
    already_segmented = check_if_done(filename_base)
    st.markdown(f"**Status:** {'✅ Already segmented' if already_segmented else '❌ Not yet segmented'}")

if prev and st.session_state.current_index > 0:
    st.session_state.current_index -= 1
if next and st.session_state.current_index < len(image_files) - 1:
    st.session_state.current_index += 1

# Load image
image_path = os.path.join(IMAGE_FOLDER, image_files[st.session_state.current_index])
image_bgr = cv2.imread(image_path)
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
H, W = image.shape[:2]

# UI controls
mode = st.radio("Mode", ["Foreground", "Background", "Eraser"], horizontal=True)
brush_size = st.slider("Brush Size", 1, 50, 10)

# Canvas for drawing
if f"scribble_mask_{filename_base}" not in st.session_state:
    st.session_state[f"scribble_mask_{filename_base}"] = np.zeros((H, W), dtype=np.uint8)

# Prepare overlay for display
overlay = np.zeros_like(image)
mask = st.session_state[f"scribble_mask_{filename_base}"]
overlay[mask == 1] = (255, 0, 0)  # Foreground: Red
overlay[mask == 2] = (0, 0, 255)  # Background: Blue

display_img = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
display_img_pil = Image.fromarray(display_img)

st.markdown(f"**Image:** {image_files[st.session_state.current_index]}")
canvas_result = st_canvas(
    fill_color="#00000000",
    stroke_width=brush_size,
    stroke_color="#ff0000" if mode == "Foreground" else ("#0000ff" if mode == "Background" else "#000000"),
    background_image=display_img_pil,
    update_streamlit=True,
    height=H,
    width=W,
    drawing_mode="freedraw",
    key=f"canvas_{filename_base}_{mode}"
)

# Update mask with new strokes
def update_mask_from_canvas(canvas_result, mode, mask):
    if canvas_result.image_data is not None:
        # Find where the user drew (non-transparent pixels)
        drawn = (canvas_result.image_data[..., 3] > 0)
        if mode == "Foreground":
            mask[drawn] = 1
        elif mode == "Background":
            mask[drawn] = 2
        elif mode == "Eraser":
            mask[drawn] = 0
    return mask

if canvas_result.image_data is not None:
    st.session_state[f"scribble_mask_{filename_base}"] = update_mask_from_canvas(canvas_result, mode, mask)

# Undo/Clear
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("↩ Undo (Clear last mode)"):
        st.session_state[f"scribble_mask_{filename_base}"] = np.zeros((H, W), dtype=np.uint8)
with col2:
    if st.button("🧹 Clear All"):
        st.session_state[f"scribble_mask_{filename_base}"] = np.zeros((H, W), dtype=np.uint8)

# Run Graph-Cut
with col3:
    if st.button("Run Graph-Cut"):
        mask = st.session_state[f"scribble_mask_{filename_base}"]
        seg_mask = graphcut(image, mask)
        seg_mask = ~seg_mask
        cutout = np.zeros_like(image)
        cutout[seg_mask] = image[seg_mask]
        mask_path = os.path.join(MASK_OUTPUT_FOLDER, f"{filename_base}_mask.png")
        cutout_path = os.path.join(CUTOUT_OUTPUT_FOLDER, f"{filename_base}_chopped.png")
        cv2.imwrite(mask_path, (seg_mask.astype(np.uint8) * 255))
        cv2.imwrite(cutout_path, cv2.cvtColor(cutout, cv2.COLOR_RGB2BGR))
        st.success(f"Saved: {mask_path}\nSaved: {cutout_path}")
        st.image([image, cutout], caption=["Original", f"{filename_base}_chopped.png"], width=256)
