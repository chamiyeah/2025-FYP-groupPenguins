import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from matplotlib.patches import Circle
from IPython.display import display
import maxflow

# Get base path and image directory
base_path = os.path.dirname(__file__)
images_path = os.path.abspath(os.path.join(base_path, "..", "data", "skin_images"))

# Load the first available image in skin_images
image_files = [f for f in os.listdir(images_path) if f.lower().endswith('.png')]
if not image_files:
    raise ValueError("No PNG images found in skin_images folder.")

image_path = os.path.join(images_path, image_files[0])
image_bgr = cv2.imread(image_path)
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Initialize scribble mask and drawing data
scribble_mask = np.zeros(image.shape[:2], dtype=np.uint8)
brush_radius = 5
current_mode = 'foreground'
overlay_data = np.zeros_like(image, dtype=np.uint8)
scribble_points = []

# Function to draw scribbles on the image
def draw_ui(x, y):
    global scribble_mask, overlay_data, scribble_points

    color = (255, 0, 0) if current_mode == 'foreground' else (0, 255, 255)
    mask_value = 1 if current_mode == 'foreground' else 2

    cv2.circle(scribble_mask, (x, y), brush_radius, mask_value, -1)
    cv2.circle(overlay_data, (x, y), brush_radius, color, -1)
    scribble_points.append((x, y, mask_value))

    overlay_image.set_data(overlay_data)
    fig.canvas.draw_idle()

# Mouse press and drag events
def on_press(event):
    if event.inaxes != ax:
        return
    draw_ui(int(event.xdata), int(event.ydata))

def on_motion(event):
    if event.inaxes == ax and event.buttons[0]:
        draw_ui(int(event.xdata), int(event.ydata))

# Run segmentation using Graph Cut
def on_segment_clicked(_):
    with output:
        print("Running Graph-Cut segmentation...")

    fg_pixels = np.where(scribble_mask == 1)
    bg_pixels = np.where(scribble_mask == 2)

    if len(fg_pixels[0]) == 0 or len(bg_pixels[0]) == 0:
        print("Please draw scribbles for both foreground and background.")
        return

    img = image.astype(np.float64)
    fg_mean = np.mean(img[fg_pixels], axis=0)
    bg_mean = np.mean(img[bg_pixels], axis=0)

    fg_dist = np.linalg.norm(img - fg_mean, axis=2)
    bg_dist = np.linalg.norm(img - bg_mean, axis=2)

    fg_prob = np.exp(-fg_dist)
    bg_prob = np.exp(-bg_dist)

    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes(img.shape[:2])
    g.add_grid_tedges(nodes, fg_prob, bg_prob)
    g.add_grid_edges(nodes, 20)

    for y, x, label in scribble_points:
        if label == 1:
            g.add_tedge(nodes[x, y], 1e10, 0)
        elif label == 2:
            g.add_tedge(nodes[x, y], 0, 1e10)

    g.maxflow()
    result_mask = g.get_grid_segments(nodes)

    seg_display = np.zeros_like(image)
    seg_display[result_mask == False] = (0, 255, 0)

    fig2, ax2 = plt.subplots()
    ax2.imshow(seg_display)
    ax2.set_title("Segmented Mask")
    ax2.axis('off')
    plt.show()

# Interface: display image and drawing tools
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image)
overlay_image = ax.imshow(overlay_data, alpha=0.5)
ax.set_title("Use red for FG, blue for BG")
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_motion)

# Toggle button to switch between foreground and background
mode_selector = widgets.ToggleButtons(
    options=[('Foreground', 'foreground'), ('Background', 'background')],
    description='Mode:'
)

def on_mode_change(change):
    global current_mode
    current_mode = change['new']

mode_selector.observe(on_mode_change, names='value')
display(mode_selector)

# Button to run Graph Cut segmentation
segment_button = widgets.Button(description="Run Graph-Cut")
output = widgets.Output()
segment_button.on_click(on_segment_clicked)

display(segment_button, output)
plt.show()
