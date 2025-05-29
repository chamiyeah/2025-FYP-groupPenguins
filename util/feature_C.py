import os
os.environ["OMP_NUM_THREADS"] = "1"
import cv2
import numpy as np
from math import nan
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from scipy.stats import circmean, circstd, entropy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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

########################## STEP 1: Downsize for faster processing ########################33
def downsizing(img, mask, downsizing_factor=0.4):
    #crucial step to improve running time
    img = cv2.resize(img, (int(img.shape[1] * downsizing_factor), int(img.shape[0] * downsizing_factor)), interpolation=cv2.INTER_AREA) #key step to improve running time
    mask = cv2.resize(mask, (int(mask.shape[1] * downsizing_factor), int(mask.shape[0] * downsizing_factor)), interpolation=cv2.INTER_NEAREST) #key step to improve running time
    return img, mask


##################### STEP 2: Get chunks of pixels for faster processing based on color similarity ##################
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


######################## STEP 3: Get the avg color in each superpixel ######################
def get_hsv_means(hsv_image, slic_segments):
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

    return np.array(hsv_means) #each entry corresponds to one superpixel (labeled form 1 to n_segments) and contains its average HSV color.



################# STEP 4: It calculates the avg HSV value across all superpixels and its std deviation, this will be key in the classification step ###################
def avg_std_hsv(hsv_means):
    hsv_array = np.array(hsv_means)  # shape (n_superpixels, 3)

    # Handle circular mean/std for Hue
    mean_h = circmean(hsv_array[:, 0], high=1, low=0, nan_policy='omit')
    std_h = circstd(hsv_array[:, 0], high=1, low=0, nan_policy='omit')

    # Linear mean/std for Saturation and Value
    mean_s = np.mean(hsv_array[:, 1])
    std_s = np.std(hsv_array[:, 1])
    mean_v = np.mean(hsv_array[:, 2])
    std_v = np.std(hsv_array[:, 2])

    return {
        'mean_H': mean_h,
        'std_H': std_h,
        'mean_S': mean_s,
        'std_S': std_s,
        'mean_V': mean_v,
        'std_V': std_v
    }



################## STEP 5: calculate entropy across all superpixels ########################
def color_entropy_from_superpixels(hsv_means, bins=10):
    hues = np.array([h[0] for h in hsv_means])
    hist, _ = np.histogram(hues, bins=bins, range=(0, 1), density=True)
    return entropy(hist + 1e-6)


#entropy acrosss pixels only (not sperpixels)
def color_entropy_from_pixels(hsv_img, mask, bins=20):
    h = hsv_img[:, :, 0][mask > 0]
    hist, _ = np.histogram(h, bins=bins, range=(0, 1), density=True)
    return entropy(hist + 1e-6)


########### STEP 6: Get a number of dominant colors in the lesion using superpixels ################3

#helper function: get a 2D matrix with rows corresponding to each superpixel: hue, sat, val, row, col (mean values + centroid position)
def get_hsv_pos_features(hsv_means, slic_segments):
    
    features = []

    for i in range(1, slic_segments.max() + 1): #considering slic_segments is a 2d array where 0's correspond with the mask, and then labels from 1 to 50 
        mask = slic_segments == i # mask for the superpixel
        coords = np.argwhere(mask)  # (row, col) positions
        centroid = coords.mean(axis=0)  # (row_center, col_center)

        hsv = hsv_means[i - 1] #get the hsv mean value for that super pixel
        feature = np.concatenate([hsv, centroid])  # make it into a 5d feature: hue, sat, val, row, col ((mean),(position))
        features.append(feature)

    return np.array(features)


#main: test different k values to get the best k (ie number of dominant colors in the image)
def count_dominant_colors(features, k_range=range(2, 11)):
    best_k = 2
    best_score = -1

    n_superpixels = features.shape[0]

    if n_superpixels < 3:
        return 1 if n_superpixels == 1 else 2 #impossible to calculate color clusters if no of superpixels is already low

    for k in k_range:
        if k >= n_superpixels: #maybe no of superpixels is already good as a color clusters measure
            continue

        kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
        labels = kmeans.labels_

        if len(set(labels)) > 1:
            try:
                score = silhouette_score(features, labels) # the silhouette score measures how well-separated and cohesive the clusters are
                if score > best_score:
                    best_score = score
                    best_k = k
            except ValueError:
                continue

    return best_k


############# STEP 7: find the number of melanoma colors according to Kasmi's approach ###############
def match_melanoma_colors_hsv(hsv_means, kasmi_threshold=0.4):
    # assigns each superpixel to a melanoma color class if the color distance is below the threshold of 0.04 that Kasmi proposes in his paper.
    ref_array = np.array(list(melanoma_hsv_colors.values()))
    color_labels = list(melanoma_hsv_colors.keys())

    matched_colors = set()

    for hsv in hsv_means:
        dh = np.minimum(np.abs(hsv[0] - ref_array[:, 0]), 1 - np.abs(hsv[0] - ref_array[:, 0])) #hue distance (circular)
        ds = np.abs(hsv[1] - ref_array[:, 1])
        dv = np.abs(hsv[2] - ref_array[:, 2])
        distances = np.sqrt((dh) ** 2 + ds ** 2 + dv ** 2)

        min_idx = np.argmin(distances)
        if distances[min_idx] < kasmi_threshold:
            matched_colors.add(color_labels[min_idx])


    return len(matched_colors) #number of melanoma colors present


############ WRAPPER FUNCTION ###########
def get_color_vector(image, mask, downsizing_factor = 0.4, n_segments = 50, compactness = 0.1):
# main function that takes an RGB!! image and its mask as an input and provides a melanoma color proportion vector
# IMPORTANT THAT THE BINARY MASK IS 2D    
# downsize first!! improves running time significantly

    if mask is None or np.all(mask == 0) or np.all(mask == 1):
        return {
            'mean_H': np.nan, 'std_H': np.nan,
            'mean_S': np.nan, 'std_S': np.nan,
            'mean_V': np.nan, 'std_V': np.nan,
            'color_entropy': np.nan, 'dominant_colors': np.nan
        }

    image, mask = downsizing(image, mask, downsizing_factor)

    slic_segments = slic_segmentation(image, mask, n_segments, compactness) #running slic segmentation on RGB not on hsv

    hsv_img = rgb2hsv(image) #getting hsv img to run future functions

    hsv_means = get_hsv_means(hsv_img, slic_segments)

    #get the feature vector values separately
    hsv_stats = avg_std_hsv(hsv_means)
    # melanoma_props = match_melanoma_colors_hsv(hsv_means, kasmi_threshold)

    result = hsv_stats
    #result['color_entropy_sup'] = color_entropy_from_superpixels(hsv_means)
    result['color_entropy'] = color_entropy_from_pixels(hsv_img, mask)
    # result['dominant_colors'] = count_dominant_colors(get_hsv_pos_features(hsv_means, slic_segments))
    result['melanoma_colors'] = match_melanoma_colors_hsv(hsv_means)

    return result