import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from util.feature_extract import (measure_pigment_network, measure_blue_veil, 
                                     measure_vascular, measure_globules, measure_streaks,
                                     measure_irregular_pigmentation, measure_regression,
                                     get_asymmetry, get_compactness, convexity_score)
from util.inpaint_util import removeHair
from util.img_util import readImageFile

def process_single_image(image_path, mask_path, output_dir):
  #Process one image with its mask and extract features.
    
    # Read image and mask
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None:
        return None, None
    
    # Convert to grayscale for hair removal
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Hair removal
    _, _, image_no_hair = removeHair(image, img_gray, kernel_size=25, threshold=10, radius=3)
    
    # Create RGBA image with transparency
    rgba = cv2.cvtColor(image_no_hair, cv2.COLOR_BGR2BGRA)
    
    # Set alpha channel based on mask
    rgba[:, :, 3] = mask
    
    # Save masked image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"masked_{filename}")
    cv2.imwrite(output_path, rgba)
    
    # Extract features
    features = {
        'image_name': filename,
        'pigment_network': measure_pigment_network(image_no_hair),
        'blue_veil': measure_blue_veil(image_no_hair),
        'vascular': measure_vascular(image_no_hair),
        'globules': measure_globules(image_no_hair),
        'streaks': measure_streaks(image_no_hair),
        'irregular_pigmentation': measure_irregular_pigmentation(image_no_hair),
        'regression': measure_regression(image_no_hair),
        'asymmetry': get_asymmetry(mask),
        'compactness': get_compactness(mask),
        'convexity': convexity_score(mask)
    }
    
    return filename, features

def process_image_dataset(input_dir, mask_dir, output_dir, n_threads=8):
   #Parrarel processing and feature extract 
    
    # output dir sanity check
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of images
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png') and not f.endswith('_mask.png')]
    
    # Name maching for image and mask
    image_paths = [os.path.join(input_dir, f) for f in image_files]
    mask_paths = [os.path.join(mask_dir, f.replace('.png', '_mask.png')) for f in image_files]
    
    features_dict = {}
    
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for img_path, mask_path in zip(image_paths, mask_paths):
            future = executor.submit(process_single_image, img_path, mask_path, output_dir)
            futures.append(future)
        
        # Collect results
        for future in futures:
            filename, features = future.result()
            if features is not None:
                features_dict[filename] = features
    
    # Pndas DF
    df = pd.DataFrame.from_dict(features_dict, orient='index')
    
    # Save features
    df.to_csv(os.path.join(output_dir, 'extracted_features.csv'), index=False)
    
    return df

if __name__ == "__main__":

    INPUT_DIR = "data\skin_images"
    MASK_DIR = "data\lesion_masks"
    OUTPUT_DIR = "data\masked_out_images"
    
    features_df = process_image_dataset(INPUT_DIR, MASK_DIR, OUTPUT_DIR)
    print(f"Processed {len(features_df)} images")