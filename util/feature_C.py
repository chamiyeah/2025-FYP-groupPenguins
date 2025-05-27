import cv2
import numpy as np
from math import nan
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from scipy.stats import circmean
from skimage.color import rgb2hsv
import numpy as np

# colors from KASMI2015CLASSIFICATION: https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/iet-ipr.2015.0385
melanoma_color_labels = ['White', 'Black', 'Red', 'Light-brown', 'Dark-brown', 'Blue-gray']
melanoma_rgb_colors = {
    'White':       [197, 188, 217],
    'Black':       [41, 31, 30],
    'Red':         [118, 21, 17],
    'Light-brown': [163, 82, 16],
    'Dark-brown':  [135, 44, 5],
    'Blue-gray':   [113, 108, 139]
}

# # Convert to HSV
# melanoma_hsv_colors = {}
# for name, rgb in melanoma_rgb_colors.items():
#     rgb_normalized = (np.array(rgb) / 255.0).reshape(1,1,3) # most HSV conversion functions assume RGB values are floating-point numbers between 0 and 1
#     hsv = rgb2hsv(rgb_normalized)[0, 0]  
#     melanoma_hsv_colors[name] = hsv

# print(melanoma_hsv_colors)

#values obtained from conversion code written above
melanoma_color_labels = ['White', 'Black', 'Red', 'Light-brown', 'Dark-brown', 'Blue-gray']
melanoma_hsv_colors = {'White': np.array([0.7183908 , 0.13364055, 0.85098039]), 
                       'Black': np.array([0.01515152, 0.26829268, 0.16078431]), 
                       'Red': np.array([0.00660066, 0.8559322 , 0.4627451 ]), 
                       'Light-brown': np.array([0.07482993, 0.90184049, 0.63921569]), 
                       'Dark-brown': np.array([0.05      , 0.96296296, 0.52941176]), 
                       'Blue-gray': np.array([0.69354839, 0.22302158, 0.54509804])}


def downsizing(img, mask, downsizing_factor=0.4):
    #crucial step to improve running time
    img = cv2.resize(img, (int(img.shape[1] * downsizing_factor), int(img.shape[0] * downsizing_factor)), interpolation=cv2.INTER_AREA) #key step to improve running time
    mask = cv2.resize(mask, (int(mask.shape[1] * downsizing_factor), int(mask.shape[0] * downsizing_factor)), interpolation=cv2.INTER_NEAREST) #key step to improve running time
    return img, mask


def slic_segmentation(image, mask, n_segments = 50, compactness = 0.1):
    #applies SLIC superpixel segmentation on the already segmented image. reduces dimensionality in the color data.
    #output is a 2D array the same size as the image, each pixel is labeled with a superpixel ID represented by an integer starting from 1.
    
    slic_segments = slic(image,
                    n_segments = n_segments,
                    compactness = compactness,
                    sigma = 0,
                    mask = mask,
                    start_label = 1,
                    channel_axis = 2)
    
    return slic_segments


# def get_rgb_means(image, slic_segments):
#     # image is assumed to be in RGB format
#     flat_segments = slic_segments.flatten()
#     flat_rgb = image.reshape(-1, 3)  

#     max_id = flat_segments.max()
#     rgb_means = []

#     for i in range(1, max_id + 1):
#         mask = flat_segments == i
#         segment = flat_rgb[mask]

#          #avoid taking extra computation time in means calculation
#         if len(segment) == 0:
#             rgb_means.append(np.array([0.0, 0.0, 0.0]))
#             continue

#         mean_rgb = np.mean(segment, axis=0)
#         rgb_means.append(mean_rgb)

#     return rgb_means  # each entry is the average RGB color of one superpixel


# def match_melanoma_colors(rgb_means, kasmi_threshold=0.4):
#     # assigns each superpixel to a melanoma color class if the color distance is below the threshold of 0.4 (Kasmi 2015)
#     rgb_array = np.array(rgb_means)  # shape: (n_segments, 3)
#     ref_array = np.array(list(melanoma_rgb_colors.values()))  # shape: (6, 3)

#     color_counts = np.zeros(len(ref_array))

#     for rgb in rgb_array:
#         distances = np.linalg.norm((rgb - ref_array) / 255.0, axis=1)  # normalize distance
#         min_idx = np.argmin(distances)
#         if distances[min_idx] < kasmi_threshold:
#             color_counts[min_idx] += 1

#     # Normalize by total number of superpixels
#     return color_counts / len(rgb_array)


def get_hsv_means(image, slic_segments):
    hsv_image = rgb2hsv(image)
    flat_segments = slic_segments.flatten()
    flat_hsv = hsv_image.reshape(-1, 3)

    max_id = flat_segments.max()
    hsv_means = []

    for i in range(1, max_id + 1):
        mask = flat_segments == i
        segment = flat_hsv[mask]

        #avoid taking extra computation time in means calculation
        if len(segment) == 0:
            hsv_means.append(np.array([0.0, 0.0, 0.0]))
            continue

        hue = circmean(segment[:, 0], high=1, low=0, nan_policy='omit')
        sat = np.mean(segment[:, 1])
        val = np.mean(segment[:, 2])

        hsv_means.append(np.array([hue, sat, val]))

    return hsv_means #each entry corresponds to one superpixel (labeled form 1 to n_segments) and contains its average HSV color.


def match_melanoma_colors_hsv(hsv_means, kasmi_threshold=0.2):
    # assigns each superpixel to a melanoma color class if the color distance is below the threshold of 0.04 that Kasmi proposes in his paper.
    hsv_array = np.array(hsv_means)  # shape: (n_segments, 3)
    ref_array = np.array(list(melanoma_hsv_colors.values()))  # shape: (6, 3)

    color_counts = np.zeros(len(ref_array))

    for hsv in hsv_array:
        # Circular-aware hue distance
        dh = np.minimum(np.abs(hsv[0] - ref_array[:, 0]), 1 - np.abs(hsv[0] - ref_array[:, 0]))
        ds = np.abs(hsv[1] - ref_array[:, 1])
        dv = np.abs(hsv[2] - ref_array[:, 2])
        distances = np.sqrt((dh) ** 2 + ds ** 2 + dv ** 2) 
        #puts more weight on hue difference than saturation and value differences, to reduce the undesirable effect of lighting

        min_idx = np.argmin(distances)
        if distances[min_idx] < kasmi_threshold:
            color_counts[min_idx] += 1

    # Normalize by total number of superpixels
    return color_counts / len(hsv_array)


def get_color_vector(image, mask, downsizing_factor = 0.4, n_segments = 50, compactness = 0.1, kasmi_threshold=0.4):
# main function that takes an RGB!! image and its mask as an input and provides a melanoma color proportion vector
# IMPORTANT THAT THE BINARY MASK IS 2D    
# downsize first!! improves running time significantly

    if mask is None or np.all(mask == 0) or np.all(mask == 1):
        return np.full(6, np.nan)

    image, mask = downsizing(image, mask, downsizing_factor)

    slic_segments = slic_segmentation(image, mask, n_segments, compactness)

    rgb_means = get_hsv_means(image, slic_segments)

    return match_melanoma_colors_hsv(rgb_means, kasmi_threshold)