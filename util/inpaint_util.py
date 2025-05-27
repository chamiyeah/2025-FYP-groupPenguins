import cv2
from feature_hair_coverage import hair_coverage
import numpy as np
import matplotlib.pyplot as plt

#hair removal function that takes original image in RGB as img_org and grayscale image as img_gray
def removeHair(img_org, img_gray, kernel_size=15, threshold=10, radius=3):
    
    #check if skin is without hair, if yes then return original picture only
    coverage = hair_coverage(img_gray)
    if coverage < 0.005:
        return img_org.copy()
    
    #adapting kernel size usin hair coverage feature
    kernel_size = 25 if coverage > 0.025 else 15
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    
    #kernel for the morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    #perform the blackHat filtering on the grayscale image to find the hair countours
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    #intensify the hair countours in preparation for the inpainting algorithm
    _, thresh = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    #inpaint the original image depending on the mask
    img_out = cv2.inpaint(img_org, thresh, radius, cv2.INPAINT_TELEA)

    return img_out
