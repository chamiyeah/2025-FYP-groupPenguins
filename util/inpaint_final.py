import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "1"

def enhance_hair_mask(img_gray):
    # contrast enhancement - CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)

    # apply sobel edge detection in both x and y directions
    edges_sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    edges_sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
    edges_sobel = np.uint8(255 * edges_sobel / np.max(edges_sobel))

    # thin hair management - laplacian
    edges_laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    edges_laplacian = np.uint8(255 * np.absolute(edges_laplacian) / np.max(np.absolute(edges_laplacian)))

    # combine both edge maps
    combined_edges = np.uint8((edges_sobel.astype(np.float32) + edges_laplacian.astype(np.float32)) / 2)
    return combined_edges

def hair_coverage(img_gray, mask):
    """Compute hair coverage ratio only within the lesion mask."""
    combined_edges = enhance_hair_mask(img_gray)
    threshold_value = int(0.3 * 255)
    hair_mask = (combined_edges > threshold_value).astype(np.uint8)

    lesion_area = np.sum(mask > 0)
    hair_on_lesion = np.logical_and(hair_mask > 0, mask > 0)
    coverage = np.sum(hair_on_lesion) / lesion_area if lesion_area > 0 else 0

    return round(coverage, 4)

def removeHair(img_org, img_gray, kernel_size=15, threshold=10, radius=3):
    # Check if skin is without hair, if yes then return original picture only
    coverage = hair_coverage(img_gray, np.ones_like(img_gray))  # use full mask for initial check
    if coverage < 0.005:
        return img_org.copy()

    img_gray = np.array(img_gray, dtype=np.uint8)
    enhanced_hair_mask = enhance_hair_mask(img_gray)

    kernel_size = 25 if coverage > 0.025 else 15
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    enhanced_hair_mask = np.array(enhanced_hair_mask, dtype=np.uint8)
    blackhat = cv2.morphologyEx(enhanced_hair_mask, cv2.MORPH_BLACKHAT, kernel, iterations=1)
    tophat = cv2.morphologyEx(enhanced_hair_mask, cv2.MORPH_TOPHAT, kernel, iterations=1)

    blackhat_float = np.array(blackhat, dtype=np.float32) * 0.6
    tophat_float = np.array(tophat, dtype=np.float32) * 0.4
    combined_mask = np.array(blackhat_float + tophat_float, dtype=np.uint8)

    adaptive_thresh = cv2.adaptiveThreshold(
        combined_mask,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    kernel_clean = np.ones((3, 3), np.uint8)
    mask_cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel_clean)

    img_out = cv2.inpaint(img_org, mask_cleaned, radius, cv2.INPAINT_TELEA)
    return img_out