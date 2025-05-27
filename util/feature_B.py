
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Boarder extraction feature using classical compactness formula, but optimized courner detection techniques
def B_compactness(mask):#where m is a gray scale mask
    #function used to find all detected borders, shapes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #extracting the largest boarder, furthest from the center with largest area
    largest_contour = max(contours, key=cv2.contourArea)
    #computes euclidean distancesfor perimeter determination, close = True for closed countour shape
    perimeter = cv2.arcLength(largest_contour, closed=True)
    #calculates area using Green’s Theorem treating contour as a polygon
    area = cv2.contourArea(largest_contour)
    #calculating compactness using classical formula
    compactness_score = (4 * np.pi * area) / (perimeter ** 2)

    return compactness_score

