import numpy as np
from math import nan
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from scipy.stats import circmean
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
melanoma_color_labels = ['White', 'Black', 'Red', 'Light-brown', 'Dark-brown', 'Blue-gray']
melanoma_hsv_colors = {'White': np.array([0.7183908 , 0.13364055, 0.85098039]), 
                       'Black': np.array([0.01515152, 0.26829268, 0.16078431]), 
                       'Red': np.array([0.00660066, 0.8559322 , 0.4627451 ]), 
                       'Light-brown': np.array([0.07482993, 0.90184049, 0.63921569]), 
                       'Dark-brown': np.array([0.05      , 0.96296296, 0.52941176]), 
                       'Blue-gray': np.array([0.69354839, 0.22302158, 0.54509804])}


def slic_segmentation(image, mask, n_segments = 50, compactness = 0.1):
    #applies SLIC superpixel segmentation on the already segmented image. reduces dimensionality in the color data.
    #output is a 2D array the same size as the image, each pixel is labeled with a superpixel ID represented by an integer starting from 1.
    
    slic_segments = slic(image,
                    n_segments = n_segments,
                    compactness = compactness,
                    sigma = 1,
                    mask = mask,
                    start_label = 1,
                    channel_axis = 2)
    
    return slic_segments


def get_hsv_means(image, slic_segments): #find the avg color in each superpixel
    
    hsv_image = rgb2hsv(image)

    max_segment_id = np.unique(slic_segments)[-1]

    hsv_means = []
    for i in range(1, max_segment_id + 1):

        segment = hsv_image.copy()
        segment[slic_segments != i] = nan

        hue_mean = circmean(segment[:, :, 0], high=1, low=0, nan_policy='omit') 
        sat_mean = np.mean(segment[:, :, 1], where = (slic_segments == i))  
        val_mean = np.mean(segment[:, :, 2], where = (slic_segments == i)) 

        hsv_mean = np.asarray([hue_mean, sat_mean, val_mean])

        hsv_means.append(hsv_mean)
        
    return hsv_means #each entry corresponds to one superpixel (labeled form 1 to 50) and contains its average HSV color.


def hsv_circular_distance(hsv1, hsv2):
    # hue: circular distance on [0,1]
    dh = min(abs(hsv1[0] - hsv2[0]), 1 - abs(hsv1[0] - hsv2[0])) #handles the wrap-around by finding the shortest arc between two hues on the color circle
    ds = abs(hsv1[1] - hsv2[1]) #non-circular, linear distance
    dv = abs(hsv1[2] - hsv2[2]) #non-circular, linear distance
    
    # hue differences are more important diagnostically than small changes in brightness or saturation, also consider
    # the effects of lighting in saturation and value which are not directly related to the skin lesion
    return np.sqrt((2 * dh)**2 + (ds**2) + (dv**2))


def match_melanoma_colors(hsv_means, kasmi_threshold=0.4):
    # assigns each superpixel to a melanoma color class if the color distance is below the threshold of 0.04 that Kasmi proposes in his paper.

    color_counts = np.zeros(6)

    for hsv in hsv_means:
        distances = [hsv_circular_distance(hsv, melanoma_hsv_colors[label]) for label in melanoma_color_labels]
        min_dist = min(distances)
        min_index = distances.index(min_dist)

        if min_dist < kasmi_threshold:
            color_counts[min_index] += 1

    # Normalize by total number of superpixels
    proportions = color_counts / len(hsv_means)

    return proportions


def get_color_vector(image, mask, n_segments = 50, compactness = 0.1, kasmi_threshold=0.4):
# main function that takes an RGB!! image and its mask as an input and provides a melanoma color proportion vector
# IMPORTANT THAT THE BINARY MASK IS 2D     

    slic_segments = slic_segmentation(image, mask, n_segments, compactness)

    hsv_means = get_hsv_means(image, slic_segments)

    return match_melanoma_colors(hsv_means, kasmi_threshold)
