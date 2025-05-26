import cv2 
import numpy as np
from math import nan
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from scipy.stats import circmean, circvar
from statistics import variance
from skimage.color import rgb2hsv
import numpy as np

# colors from KASMI2015CLASSIFICATION: https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/iet-ipr.2015.0385
# melanoma_rgb_colors = {
#     'White':       [197, 188, 217],
#     'Black':       [41, 31, 30],
#     'Red':         [118, 21, 17],
#     'Light-brown': [163, 82, 16],
#     'Dark-brown':  [135, 44, 5],
#     'Blue-gray':   [113, 108, 139]
# }

# # Convert to HSV
# melanoma_hsv_colors = {}
# for name, rgb in melanoma_rgb_colors.items():
#     rgb_normalized = (np.array(rgb) / 255.0).reshape(1,1,3) # most HSV conversion functions assume RGB values are floating-point numbers between 0 and 1
#     hsv = rgb2hsv(rgb_normalized)[0, 0]  
#     melanoma_hsv_colors[name] = hsv

# print(melanoma_hsv_colors)

#values obtained from conversion code written above
melanoma_hsv_colors = {'White': np.array([0.7183908 , 0.13364055, 0.85098039]), 
                       'Black': np.array([0.01515152, 0.26829268, 0.16078431]), 
                       'Red': np.array([0.00660066, 0.8559322 , 0.4627451 ]), 
                       'Light-brown': np.array([0.07482993, 0.90184049, 0.63921569]), 
                       'Dark-brown': np.array([0.05      , 0.96296296, 0.52941176]), 
                       'Blue-gray': np.array([0.69354839, 0.22302158, 0.54509804])}


def crop_masked(masked_img): 
    #crops the image tightly around lesion masks removing excess of blak bg pixels. it avoids skewing the color statistics.
    
    active_img = np.sum(masked_img, axis=2)

    active_rows = np.any(active_img != 0, axis=1)
    active_cols = np.any(active_img != 0, axis=0)
    
    row_min, row_max = np.where(active_rows)[0][[0, -1]]
    col_min, col_max = np.where(active_cols)[0][[0, -1]]

    return masked_img[row_min:row_max+1, col_min:col_max+1]

# Load image using cv2
img_path = ".\masked_out_lesions\PAT_566_178_625_chopped.png"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed

# Crop
cropped = crop_masked(img)
cv2.imshow(cropped)

# def slic_segmentation(image, mask, n_segments = 50, compactness = 0.1):
#     #applies SLIC superpixel segmentation on the already segmented image. reduces dimensionality in the color data.
#     #output is a 2D array the same size as the image, each pixel is labeled with a superpixel ID represented by an integer starting from 1.
    
#     slic_segments = slic(image,
#                     n_segments = n_segments,
#                     compactness = compactness,
#                     sigma = 1,
#                     mask = mask,
#                     start_label = 1,
#                     channel_axis = 2)
    
#     return slic_segments

# def get_hsv_means(image, slic_segments): #find the avg color in each superpixel
    
#     hsv_image = rgb2hsv(image)

#     max_segment_id = np.unique(slic_segments)[-1]

#     hsv_means = []
#     for i in range(1, max_segment_id + 1):

#         segment = hsv_image.copy()
#         segment[slic_segments != i] = nan

#         hue_mean = circmean(segment[:, :, 0], high=1, low=0, nan_policy='omit') 
#         sat_mean = np.mean(segment[:, :, 1], where = (slic_segments == i))  
#         val_mean = np.mean(segment[:, :, 2], where = (slic_segments == i)) 

#         hsv_mean = np.asarray([hue_mean, sat_mean, val_mean])

#         hsv_means.append(hsv_mean)
        
#     return hsv_means


# def color_dominance(image, mask, clusters = 5, include_ratios = False): #by using kMeans we extract dominant colors in HSV space 
    
#     cut_im = cut_im_by_mask(image, mask) 
#     hsv_im = rgb2hsv(cut_im) 
#     flat_im = np.reshape(hsv_im, (-1, 3)) 

#     k_means = KMeans(n_clusters=clusters, n_init=10, random_state=0)
#     k_means.fit(flat_im)

#     dom_colors = np.array(k_means.cluster_centers_, dtype='float32') 

#     if include_ratios:

#         counts = np.unique(k_means.labels_, return_counts=True)[1] 
#         ratios = counts / flat_im.shape[0] 

#         r_and_c = zip(ratios, dom_colors) 
#         r_and_c = sorted(r_and_c, key=lambda x: x[0],reverse=True) 

#         return r_and_c
    
#     return dom_colors


# def get_melanoma_color_vector(image, mask, threshold=0.05, distance_cutoff=0.4):
#     """
#     Computes a 1x6 binary vector indicating presence of melanoma-related colors
#     based on superpixel-wise HSV color matching.

#     Parameters:
#         image (np.array): RGB image
#         mask (np.array): Binary mask of the lesion
#         threshold (float): Minimum proportion of superpixels required to mark a color as 'present'
#         distance_cutoff (float): Maximum HSV distance to consider a superpixel as matching a reference color

#     Returns:
#         np.array: 1x6 binary vector [white, black, red, light-brown, dark-brown, blue-gray]
#     """

#     # Crop image to mask
#     cut_img = cut_im_by_mask(image, mask)
#     cut_mask = cut_im_by_mask(mask, mask)  # mask also needs to be cropped

#     # Segment into superpixels
#     segments = slic_segmentation(cut_img, cut_mask)

#     # Get HSV means for each superpixel
#     hsv_means = get_hsv_means(cut_img, segments)

#     # Initialize match counts
#     color_counts = {color: 0 for color in melanoma_hsv_colors.keys()}
#     total_superpixels = len(hsv_means)

#     # Compare each superpixel HSV to each melanoma color
#     for hsv in hsv_means:
#         for name, ref_hsv in melanoma_hsv_colors.items():
#             dist = np.linalg.norm(hsv - ref_hsv)
#             if dist < distance_cutoff:
#                 color_counts[name] += 1
#                 break  # assume one color match per superpixel is enough

#     # Generate binary vector
#     binary_vector = []
#     for name in melanoma_hsv_colors.keys():
#         proportion = color_counts[name] / total_superpixels
#         binary_vector.append(1 if proportion >= threshold else 0)

#     return np.array(binary_vector)

# def get_melanoma_color_vector(image, mask, threshold=0.05, distance_cutoff=0.4):
#     """
#     Computes a 1x6 binary vector indicating presence of melanoma-related colors
#     based on superpixel-wise HSV color matching.

#     Parameters:
#         image (np.array): RGB image
#         mask (np.array): Binary mask of the lesion
#         threshold (float): Minimum proportion of superpixels required to mark a color as 'present'
#         distance_cutoff (float): Maximum HSV distance to consider a superpixel as matching a reference color

#     Returns:
#         np.array: 1x6 binary vector [white, black, red, light-brown, dark-brown, blue-gray]
#     """
#     # Crop image to mask (optional but usually helps avoid border superpixels)
#     cut_img = cut_im_by_mask(image, mask)
#     cut_mask = cut_im_by_mask(mask, mask)  # mask also needs to be cropped

#     # Segment into superpixels
#     segments = slic_segmentation(cut_img, cut_mask)

#     # Get HSV means for each superpixel
#     hsv_means = get_hsv_means(cut_img, segments)

#     # Initialize match counts
#     color_counts = {color: 0 for color in melanoma_hsv_colors.keys()}
#     total_superpixels = len(hsv_means)

#     # Compare each superpixel HSV to each melanoma color
#     for hsv in hsv_means:
#         for name, ref_hsv in melanoma_hsv_colors.items():
#             dist = np.linalg.norm(hsv - ref_hsv)
#             if dist < distance_cutoff:
#                 color_counts[name] += 1
#                 break  # assume one color match per superpixel is enough

#     # Generate binary vector
#     binary_vector = []
#     for name in melanoma_hsv_colors.keys():
#         proportion = color_counts[name] / total_superpixels
#         binary_vector.append(1 if proportion >= threshold else 0)

#     return np.array(binary_vector)
