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

