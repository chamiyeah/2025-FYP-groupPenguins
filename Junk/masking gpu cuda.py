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
import cupy as cp  # For GPU processing
import torch  # For GPU detection

def check_gpu_availability():
    if torch.cuda.is_available():
        return True, torch.cuda.get_device_name(0)
    return False, None

def process_single_image_gpu(image_path, mask_path, output_dir):
    #Process one image with GPU acceleration
    try:
        # Read image and mask
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            return None, None
        
        # Transfer to GPU
        image_gpu = cp.asarray(image)
        mask_gpu = cp.asarray(mask)
        
        # Convert to grayscale for hair removal
        img_gray_gpu = cp.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
        # Hair removal (keeping on CPU as removeHair uses CPU-based OpenCV functions)
        _, _, image_no_hair = removeHair(cp.asnumpy(image_gpu), cp.asnumpy(img_gray_gpu))
        
        # Jump to GPU for rest
        image_no_hair_gpu = cp.asarray(image_no_hair)
        
        # Create RGBA image with transparency
        rgba = cv2.cvtColor(cp.asnumpy(image_no_hair_gpu), cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = cp.asnumpy(mask_gpu)
        
        # Save masked image
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"masked_{filename}")
        cv2.imwrite(output_path, rgba)
        
        # Extract features (convert to CPU for feature extraction functions)
        features = {
            'image_name': filename,
            'pigment_network': float(measure_pigment_network(cp.asnumpy(image_no_hair_gpu))),
            'blue_veil': float(measure_blue_veil(cp.asnumpy(image_no_hair_gpu))),
            'vascular': float(measure_vascular(cp.asnumpy(image_no_hair_gpu))),
            'globules': float(measure_globules(cp.asnumpy(image_no_hair_gpu))),
            'streaks': float(measure_streaks(cp.asnumpy(image_no_hair_gpu))),
            'irregular_pigmentation': float(measure_irregular_pigmentation(cp.asnumpy(image_no_hair_gpu))),
            'regression': float(measure_regression(cp.asnumpy(image_no_hair_gpu))),
            'asymmetry': float(get_asymmetry(cp.asnumpy(mask_gpu))),
            'compactness': float(get_compactness(cp.asnumpy(mask_gpu))),
            'convexity': float(convexity_score(cp.asnumpy(mask_gpu)))
        }
        
        # Clear GPU memory
        cp.get_default_memory_pool().free_all_blocks()
        
        return filename, features
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, None

def process_image_dataset(input_dir, mask_dir, output_dir, use_gpu=False, n_threads=8):
    #Process images with option for GPU
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png') and not f.endswith('_mask.png')]
    
    # Name matching for image and mask
    image_paths = [os.path.join(input_dir, f) for f in image_files]
    mask_paths = [os.path.join(mask_dir, f.replace('.png', '_mask.png')) for f in image_files]
    
    features_dict = {}

    
    # Select processing function based on GPU availability
    process_func = process_single_image_gpu if use_gpu else process_single_image
    
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for img_path, mask_path in zip(image_paths, mask_paths):
            future = executor.submit(process_func, img_path, mask_path, output_dir)
            futures.append(future)
        
        # Collect results
        for future in futures:
            filename, features = future.result()
            if features is not None:
                features_dict[filename] = features
    
    # DataFrame
    df = pd.DataFrame.from_dict(features_dict, orient='index')
    df.to_csv(os.path.join(output_dir, 'extracted_features.csv'), index=False)
    
    return df

if __name__ == "__main__":
    INPUT_DIR = "data\skin_images"
    MASK_DIR = "data\lesion_masks"
    OUTPUT_DIR = "data\masked_out_images"
    
    # Check GPU availability
    gpu_available, gpu_name = check_gpu_availability()
    
    if gpu_available:
        print(f"GPU detected: {gpu_name}")
        use_gpu = input("Do you want to use GPU acceleration? (y/n): ").lower() == 'y'
    else:
        print("No GPU detected. Using CPU processing.")
        use_gpu = False
    
    # Process images
    features_df = process_image_dataset(INPUT_DIR, MASK_DIR, OUTPUT_DIR, use_gpu=use_gpu)
    print(f"Processed {len(features_df)} images")