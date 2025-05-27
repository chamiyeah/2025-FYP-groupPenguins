import cv2
import numpy as np
from math import floor, ceil
from skimage.transform import rotate

#Main call -> mean_asymmetry

"""Calculating asymetry score by rotating image 4 times(90 degrees) and calculating
asymetry each time"""

def cut_mask(mask):
    """Removing big part of black background to avoid misleading 
    asymetry and computational cost"""
    rows = np.any(mask, axis=1) #rows that have white pixels
    cols = np.any(mask, axis=0) #columns that have white pixels

    row_min, row_max = np.where(rows)[0][[0, -1]] #indexes of white rows
    col_min, col_max = np.where(cols)[0][[0, -1]] #indexes of white columns

    return mask[row_min:row_max+1, col_min:col_max+1] #bounding box of the lesion

def find_midpoint(image):
    """Mid point detection of the lesion, where image is grayscale mask"""
    return image.shape[0] // 2, image.shape[1] // 2 #middle row, middle column (x,y)

def asymmetry(mask):
    """Measuring shape asymetry of the mask"""
    row_mid, col_mid = find_midpoint(mask) #middle point

    #splitting the mask into 4 halves
    upper = mask[:row_mid, :]
    lower = mask[-row_mid:, :]
    left = mask[:, :col_mid]
    right = mask[:, -col_mid:]

    #flipping bottom and right half
    flipped_lower = np.flipud(lower) #flip up
    flipped_right = np.fliplr(right) #flip left

    #the following codes are used to ensure that when halves are compared,
    #the arrays have the same sizes for np.logical_xor() to work
    hmin = min(upper.shape[0], flipped_lower.shape[0]) #picks the smallest number of rows
    vmin = min(left.shape[1], flipped_right.shape[1]) #picks the smallest number of columns

    #counting mismatched pixels
    hori_xor = np.logical_xor(upper[:hmin, :], flipped_lower[:hmin, :])
    vert_xor = np.logical_xor(left[:, :vmin], flipped_right[:, :vmin])

    #calculating the area -> sum of the white pixels
    total = np.sum(mask)
    if total == 0: #if mask is black to avoid division by zero error
        return np.nan

    #calculating the asymetry score (0 perfect symetry, 1 very asymetric)
    score = (np.sum(hori_xor) + np.sum(vert_xor)) / (2 * total) 
    #total mismatch standardized by total pixels in lesion tp get score 0 to 1
    return round(score, 4)

def rotation_asymmetry(mask, n=4):
    """Rotating the picture n times and getting asymetry 
    scores for every rotation"""
    asymmetry_scores = []

    for i in range(n):
        deg = 360 * i / n
    #optimization, if the angle is multiple of 90 use np.rot90 which is faster than rotate function
        if deg % 90 == 0:
            k = int(deg // 90)
            rotated = np.rot90(mask, k)
        else:
            rotated = rotate(mask, deg, preserve_range=True, order=0).astype(np.uint8)

        cropped = cut_mask(rotated > 0)#creating a binary bounded lesion, avoiding interpolation artifacts
        score = asymmetry(cropped)
        asymmetry_scores.append(score)

    return asymmetry_scores

def mean_asymmetry(mask, rotations=4):

    if mask is None: #check if mask exists
        return np.nan
    mask = (mask > 0).astype(np.uint8) #binarizing because mask is in gray scale

    scores = rotation_asymmetry(mask, rotations)
    return round(np.nanmean(scores), 4)
