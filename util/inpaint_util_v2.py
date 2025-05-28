import cv2
import numpy as np
from feature_hair_coverage import hair_coverage
import matplotlib.pyplot as plt

'''
this function will remove hair with any color, fixing the issue on our 1st version.
implemented Sobel and Laplacian edge detectin to ifind hair regardless of color
combined blackhat and tophat to find both black and white hair
added contrast enhancements using CLAHE (Contrast Limited Adaptive Histogram Equalization)

figured out adaptie thrusholding finally !!!

'''


def enhance_hair_mask(img_gray):
    # contrast enhancement - CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img_gray)
    
    # sobel and leplacian edge detection
    
    # apply sobel edge detection in both x and y directions
    #convert to float for more precision
    edges_sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    edges_sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
    edges_sobel = np.uint8(255 * edges_sobel / np.max(edges_sobel))
    
    #thin hair management - leplacian
    edges_laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    edges_laplacian = np.uint8(255 * np.absolute(edges_laplacian) / np.max(np.absolute(edges_laplacian)))
    
    #combine edges from sobel and laplacian
    #normalize and combine the two edge maps
    combined_edges = np.uint8((np.array(edges_sobel, dtype=np.float32) + 
                             np.array(edges_laplacian, dtype=np.float32)) / 2)
    
    return combined_edges

def removeHair(img_org, img_gray, kernel_size=15, threshold=10, radius=3):
    #check if skin is without hair, if yes then return original picture only
    coverage = hair_coverage(img_gray)
    if coverage < 0.005:
        return img_org.copy()
    
    #sanity check for imput images are infaact numpy arrays
    img_gray = np.array(img_gray, dtype=np.uint8)
    
    #enhance the hair detection using multiple techniques
    #input image is in grayscale format
    enhanced_hair_mask = enhance_hair_mask(img_gray)
    
    #adapt kernel size based on hair coverage
    kernel_size = 25 if coverage > 0.025 else 15
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    
    #morphological operations to isolate hair structures
    enhanced_hair_mask = np.array(enhanced_hair_mask, dtype=np.uint8)
    blackhat = cv2.morphologyEx(enhanced_hair_mask, cv2.MORPH_BLACKHAT, kernel, iterations=1)
    tophat = cv2.morphologyEx(enhanced_hair_mask, cv2.MORPH_TOPHAT, kernel, iterations=1)
    
    #combined both morphological operations to find both dark and light hair
    #scaled the results to enhance visibility
    blackhat_float = np.array(blackhat, dtype=np.float32) * 0.6
    tophat_float = np.array(tophat, dtype=np.float32) * 0.4
    combined_mask = np.array(blackhat_float + tophat_float, dtype=np.uint8)
    
    #dynamic thresholding based on local image statistics
    #used adaptive thresholding to create a binary mask
    # combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
    # combined_mask = cv2.medianBlur(combined_mask, 5)
    # combined_mask = cv2.bilateralFilter(combined_mask, 9, 75, 75)
    
    combined_mask = np.array(combined_mask, dtype=np.uint8)
    adaptive_thresh = cv2.adaptiveThreshold(
        combined_mask,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    #cleanup mask
    #used morphological closing to remove small holes and noise
    kernel_clean = np.ones((3,3), np.uint8)
    mask_cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel_clean)
    
    #sanity check for mask is proper for inpainting
    mask_cleaned = np.array(mask_cleaned, dtype=np.uint8)
    
    #inpaint the original image using the enhanced mask
    img_out = cv2.inpaint(img_org, mask_cleaned, radius, cv2.INPAINT_TELEA)
    
    return img_out