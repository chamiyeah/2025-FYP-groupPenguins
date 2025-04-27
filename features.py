import cv2
import os
import pandas as pd
import numpy as np
from util.img_util import readImageFile, saveImageFile, ImageDataLoader
from util.inpaint_util import removeHair
from util.feature_extract import *
#applying masks

image_dir = './result/hair_removed/'
mask_dir = './data/lesion_masks/'
output_dir = './result/masked_out_images/'

#load files from medical images folder


files = ImageDataLoader(image_dir)
image_list = files.file_list

extracted = []

for name in image_list:

    path = os.path.join(image_dir, name) 
    name_clean = name.replace('output_', '').replace('.png', '')
    base_name = os.path.splitext(name_clean)[0] #PAT_8_15_820.png.jpg without .png.jpg
    
    #finding masks for images in lesion_masks folder
    mask_name = f"{base_name}_mask.png"
    mask_path = os.path.join(mask_dir, mask_name)

    if not os.path.exists(mask_path):
        print(f"Mask not found for {name_clean}")
        continue

    img = cv2.imread(path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 
    # Apply the mask (keep only the lesion)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    #save
    output_name = f"{base_name}_chopped.png"
    output_path = os.path.join(output_dir, output_name)
    success = cv2.imwrite(output_path, masked_img)

    #extracting features
    #Asymetry
    features = {
    'image': base_name,
    'asymmetry':round(mean_asymmetry(mask), 3),
    'compactness': round(get_compactness(mask), 3),
    'convexity': round(convexity_score(mask), 3),
    'pigmentation':round(measure_irregular_pigmentation(img), 3),
    'vascularity': round(measure_vascular(img), 3)
    }

    extracted.append(features)


df = pd.DataFrame(extracted)
feature_names = ['image','asymmetry', 'compactness', 'convexity', 'irregular_pigmentation', 'vascular_measure']
df.columns = feature_names
path_csv = './result/features.csv'
df.to_csv(path_csv, index=False)
print(f"Features extracted and saved to results and chopped leasion images saved to masked_out_images")






