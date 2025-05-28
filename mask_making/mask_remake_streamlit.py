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

# --- UI COLOR HELPER ---
def get_canvas_color(mode):
    if mode == "Foreground":
        return "#ff0000"  # Red
    elif mode == "Background":
        return "#0000ff"  # Blue
    elif mode == "Eraser":
        return "#ffffff"  # White (erase)
    return "#000000"

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

# Navigation logic (move this before UI)
if 'nav_action' not in st.session_state:
    st.session_state['nav_action'] = None

def go_prev():
    if st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.session_state['nav_action'] = 'prev'

def go_next():
    if st.session_state.current_index < len(image_files) - 1:
        st.session_state.current_index += 1
        st.session_state['nav_action'] = 'next'

filename_base = os.path.splitext(image_files[st.session_state.current_index])[0]
already_segmented = check_if_done(filename_base)

# Load image
image_path = os.path.join(IMAGE_FOLDER, image_files[st.session_state.current_index])
image_bgr = cv2.imread(image_path)
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
H, W = image.shape[:2]

# Canvas mask state
if f"scribble_mask_{filename_base}" not in st.session_state:
    st.session_state[f"scribble_mask_{filename_base}"] = np.zeros((H, W), dtype=np.uint8)
if f"scribble_history_{filename_base}" not in st.session_state:
    st.session_state[f"scribble_history_{filename_base}"] = []
mask = st.session_state[f"scribble_mask_{filename_base}"]

# Prepare overlay for display
overlay = np.zeros_like(image)
overlay[mask == 1] = (255, 0, 0)  # Foreground: Red
overlay[mask == 2] = (0, 0, 255)  # Background: Blue
display_img = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
display_img_pil = Image.fromarray(display_img)

# Layout: image/canvas on the left, controls and results on the right
left_col, right_col = st.columns([2, 1], gap="small")

with left_col:
    st.markdown(f"**Image:** {image_files[st.session_state.current_index]}")
    canvas_result = st_canvas(
        fill_color="#00000000",
        stroke_width=st.session_state.get('brush_size', 10),
        stroke_color=get_canvas_color(st.session_state.get('mode', 'Foreground')),
        background_image=display_img_pil,
        update_streamlit=True,
        height=H,
        width=W,
        drawing_mode="freedraw",
        key=f"canvas_{filename_base}_{st.session_state.get('mode', 'Foreground')}"
    )

with right_col:
    st.markdown(f"### Controls")
    nav_col1, nav_col2 = st.columns([1, 1], gap="small")
    with nav_col1:
        prev = st.button("← Previous Image", key="prev_btn_right", on_click=go_prev)
    with nav_col2:
        next = st.button("→ Next Image", key="next_btn_right", on_click=go_next)
    filename_base = os.path.splitext(image_files[st.session_state.current_index])[0]
    already_segmented = check_if_done(filename_base)
    st.markdown(f"**Status:** {'✅ Already segmented' if already_segmented else '❌ Not yet segmented'}")
    st.session_state['mode'] = st.radio("Mode", ["Foreground", "Background", "Eraser"], horizontal=True, key="mode_radio", index=["Foreground", "Background", "Eraser"].index(st.session_state.get('mode', 'Foreground')) if 'mode' in st.session_state else 0)
    st.session_state['brush_size'] = st.slider("Brush Size", 1, 50, st.session_state.get('brush_size', 10), key="brush_slider")
    btn_col1, btn_col2, btn_col3 = st.columns([1,1,1], gap="small")
    with btn_col1:
        undo_clicked = st.button("↩ Undo (Clear last mode)", key="undo_btn")
    with btn_col2:
        clear_clicked = st.button("🧹 Clear All", key="clear_btn")
    with btn_col3:
        run_clicked = st.button("Run Graph-Cut", key="run_btn")

    # Output image comparison after buttons, with no gap and correct order
    if run_clicked:
        mask = st.session_state[f"scribble_mask_{filename_base}"]
        seg_mask = graphcut(image, mask)
        cutout = np.zeros_like(image)
        cutout[seg_mask] = image[seg_mask]
        mask_path = os.path.join(MASK_OUTPUT_FOLDER, f"{filename_base}_mask.png")
        cutout_path = os.path.join(CUTOUT_OUTPUT_FOLDER, f"{filename_base}_chopped.png")
        cv2.imwrite(mask_path, ((~seg_mask).astype(np.uint8) * 255))  # Save the inverted mask for compatibility
        cv2.imwrite(cutout_path, cv2.cvtColor(cutout, cv2.COLOR_RGB2BGR))
        st.success(f"Saved: {mask_path}\nSaved: {cutout_path}")
        st.markdown("#### Segmentation Result")
        out_col1, out_col2 = st.columns(2, gap="small")
        with out_col1:
            st.image(cutout, caption=f"{filename_base}_chopped.png", use_column_width=True)
        with out_col2:
            st.image(image, caption="Original", use_column_width=True)

    # Handle Undo/Clear
    if undo_clicked:
        # Undo: restore previous mask from history if available
        history = st.session_state[f"scribble_history_{filename_base}"]
        if history:
            st.session_state[f"scribble_mask_{filename_base}"] = history.pop()
        else:
            st.session_state[f"scribble_mask_{filename_base}"] = np.zeros((H, W), dtype=np.uint8)
        st.experimental_rerun()
    if clear_clicked:
        # Clear: push current mask to history, then clear
        st.session_state[f"scribble_history_{filename_base}"].append(st.session_state[f"scribble_mask_{filename_base}"].copy())
        st.session_state[f"scribble_mask_{filename_base}"] = np.zeros((H, W), dtype=np.uint8)
        st.experimental_rerun()

# Update mask with new strokes
def update_mask_from_canvas(canvas_result, mode, mask):
    if canvas_result.image_data is not None:
        drawn = (canvas_result.image_data[..., 3] > 0)
        # Only update if something was drawn
        if np.any(drawn):
            # Save current mask to history before updating
            history = st.session_state[f"scribble_history_{filename_base}"]
            history.append(mask.copy())
            if mode == "Foreground":
                mask[drawn] = 1
            elif mode == "Background":
                mask[drawn] = 2
            elif mode == "Eraser":
                mask[drawn] = 0
    return mask

if canvas_result.image_data is not None:
    st.session_state[f"scribble_mask_{filename_base}"] = update_mask_from_canvas(canvas_result, st.session_state['mode'], mask)
