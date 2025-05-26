import numpy as np
import cv2
import matplotlib.pyplot as plt

def hair_coverage(img_org, visual = False): #img_org is a grayscale image

    #reduces image noise, skin texture, and small bumps before edge detection
    blurred = cv2.GaussianBlur(img_org, (7,7), 0) 

    #laplacian operator highlights areas of rapid intensity change -> edge detection
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3) #cv2.CV_64F sets the data type to 64-bit float so it can hold both positive and negative values
    laplacian_abs = np.absolute(laplacian) #remove negative edges, because we don't care about direction (going from light to dark or vice versa), we care about magnitute
    laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) #scaling edges to 0–255 for thresholding

    threshold_value = 0.3 * 255
    hair_mask = (laplacian_norm > threshold_value).astype(np.uint8)

    #calculating hair coverage
    hair_pixels = np.sum(hair_mask)
    total_pixels = hair_mask.size
    coverage = hair_pixels / total_pixels

    return round(coverage, 4)