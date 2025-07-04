import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))
sys.path.append(str(Path(__file__).parent.parent.resolve()))

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "1"

def enhance_hair_mask(img_gray):
    """
    Enhances and detects hair-like structures and detects both strong edges (Sobel) and fine details (Laplacian)

    Input:
        img_gray (numpy.ndarray): input grayscale image

    Returns:
        numpy.ndarray: binary mask highlighting hair, with values normalized to 0-255 range

    Note:
        The function uses both Sobel and Laplacian operators as they complement each other:
        - Sobel is good at detecting strong edges and directional changes
        - Laplacian is effective at finding fine details and thin structures like hair
    """
    #enhancing contract
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)

    #detecting strong edges in both directions
    edges_sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    edges_sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
    edges_sobel = np.uint8(255 * edges_sobel / np.max(edges_sobel))

    #finding fine hair details
    edges_laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    edges_laplacian = np.uint8(255 * np.absolute(edges_laplacian) / np.max(np.absolute(edges_laplacian))) #normalizing

    #taking average of both methods and returning combined mask that highlights hair-like structures
    combined_edges = np.uint8((edges_sobel.astype(np.float32) + edges_laplacian.astype(np.float32)) / 2)
    return combined_edges

def hair_coverage(img_gray, mask):
    """
    - Calculates the ratio of hair coverage on in the lesion area 
    - Computes the ratio of hair pixels to total lesion area

    Input:
        img_gray (numpy.ndarray): grayscale 
        mask (numpy.ndarray): binary mask 

    Returns:
        float: ratio of hair coverage within the lesion area
               0 if mask is black, invalid or there is no lesion
    """
    combined_edges = enhance_hair_mask(img_gray)
    threshold_value = int(0.3 * 255) #0.3 tested value
    hair_mask = (combined_edges > threshold_value).astype(np.uint8)

    lesion_area = np.sum(mask > 0) #accounting for lesion area
    hair_on_lesion = np.logical_and(hair_mask > 0, mask > 0)
    coverage = np.sum(hair_on_lesion) / lesion_area if lesion_area > 0 else 0 #check if division by zero

    return round(coverage, 4)

def removeHair(img_org, img_gray, kernel_size=15, threshold=10, radius=3):
    """
    - Checks if hair removal is necessary (returns original photo if hair coverage < 0.5%)
    - Enhances the hair mask using enhance_hair_mask
    - Adjusts kernel size based on hair coverage
    - Uses blackhat and tophat to detect hair (combined)
    - Creates a final mask through adaptive thresholding and cleaning with small kernel
    - Inpainting algorithm removes the detected hair

    Input:
        img_org (numpy.ndarray): color image
        img_gray (numpy.ndarray): grayscale image
        kernel_size (int, optional): defaults kernel size is 15, but automatically adjusts to 25 for heavy hair (>2.5% coverage, checked by plotting with annotations)
        threshold (int, optional): defaults to 10
        radius (int, optional): defaults to 3

    Returns:
        numpy.ndarray: image with hair removed, if no hair is detected (coverage < 0.5%), returns a copy of the original image, optimizing running time
    """
    
    coverage = hair_coverage(img_gray, np.ones_like(img_gray))  # use full mask for initial check
    if coverage < 0.005:#check for hair (optimizing time)
        return img_org.copy()

    img_gray = np.array(img_gray, dtype=np.uint8)
    enhanced_hair_mask = enhance_hair_mask(img_gray) #enhance hair recognition

    kernel_size = 25 if coverage > 0.035 else 15 #adjust kernel size to avoid blurrying, kerenel size 15 is tested to be optimal for thin hair
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    enhanced_hair_mask = np.array(enhanced_hair_mask, dtype=np.uint8)
    blackhat = cv2.morphologyEx(enhanced_hair_mask, cv2.MORPH_BLACKHAT, kernel, iterations=1)
    tophat = cv2.morphologyEx(enhanced_hair_mask, cv2.MORPH_TOPHAT, kernel, iterations=1)

    #weighing blackhat and tophat masks, we chose 60/40 ratio based on testing
    blackhat_float = np.array(blackhat, dtype=np.float32) * 0.6
    tophat_float = np.array(tophat, dtype=np.float32) * 0.4
    combined_mask = np.array(blackhat_float + tophat_float, dtype=np.uint8)

    adaptive_thresh = cv2.adaptiveThreshold(combined_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    #making hair mask more accurate, (3,3) is small enough to preserve hair
    kernel_clean = np.ones((3, 3), np.uint8) 
    mask_cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel_clean)

    #removing hair, radius is 3 because it is optimal, otherwise it blurs the image
    img_out = cv2.inpaint(img_org, mask_cleaned, radius, cv2.INPAINT_TELEA)
    return img_out