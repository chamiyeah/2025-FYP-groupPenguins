import cv2
import os
import pandas as pd
import numpy as np
from util.img_util import readImageFile, saveImageFile, ImageDataLoader

image_dir = './data/skin_images/part_3/'
mask_dir = './data/lesion_masks/'
output_dir = './masked_3/'

files = ImageDataLoader(image_dir)
image_list = files.file_list

_not_found = 0

for name in image_list:

    path = os.path.join(image_dir, name) 
    base_name = os.path.splitext(name)[0] #PAT_8_15_820.png without .png

    #finding masks for images in lesion_masks folder
    mask_name = f"{base_name}_mask.png"
    mask_path = os.path.join(mask_dir, mask_name)

    if not os.path.exists(mask_path):
        print(f"Mask not found for {name}")
        _not_found+=1
        continue

    img = cv2.imread(path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 
    # Apply the mask (keep only the lesion)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    #save
    output_name = f"{base_name}_chopped.png"
    output_path = os.path.join(output_dir, output_name)
    success = cv2.imwrite(output_path, masked_img)






