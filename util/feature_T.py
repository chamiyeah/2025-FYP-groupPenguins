import cv2
import numpy as np

def mean_gradient(image, mask):
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    masked_gray = np.where(mask > 0, gray_img, 0)


    sobelx = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=5) # tried kernel sizes of 3, 5, 7 but the one that allows the best separability is ksize=5
    sobely = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=5)
    
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Rest of your code
    gradient_magnitude = gradient_magnitude[mask > 0]
    mean_gradient = np.mean(gradient_magnitude)
    
    return mean_gradient