#Set OpenMP threads to 1 to prevent nested parallelization conflicts when using multiprocessing !!!
# this is important cz numpy and OpenCV operations might try to parallle internally.
import os
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count #multiprocessing imports for parallel processing optimization
from functools import partial #partial for efficient function parameter binding in parallel processing
from pathlib import Path  #pathlib for more efficient and cross-platform file path handling
# Added tqdm for progress tracking
from tqdm import tqdm
from feature_A import mean_asymmetry
from feature_B import B_compactness
from feature_C import get_color_vector
from image_preprocessing import enhance_color_hsv_clahe


def process_single_image(row_data, image_dir, mask_dir):
    """
    Process a single image to extract features. #This has to be done if we want to use parallel processing effectively.
    
    Optimizations :
    1. function signature will accept tuple of data instead of individual parameters
    2. used more efficient path joining with os.path.join
    3. easier return structure for parallel processing
    """
    #unpack the row data tuple for parallel processing
    img_id, patient_id, cancer = row_data
    
    # file path improvements using os.path.join instead of string concatenation
    img_path = os.path.join(image_dir, img_id)
    mask_filename = img_id.replace('.png', '_mask.png')
    mask_path = os.path.join(mask_dir, mask_filename)
    
    try:
        #super efficent image reading with direct flags
        #removed multiple redundant image checks and simplified error handling
        img = cv2.imread(img_path)
        if img is None:
            print(f"img None {img_id}")
            return None
            
        #convert color space once and store result for later use instade of doing it again
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = enhance_color_hsv_clahe(img)
        
        #read mask with specific flag for grayscale to avoid redundant conversion
        #removed unnecessary checks for mask existence, as it will be handled in the try-except block
        # mask_path = os.path.join(mask_dir, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"mask None {img_id}")
            return None

        # Feature extraction remains unchanged !! It will extrack as we made it before.
        # but moved inside try-except for better error handling
        # extract features from the image and mask
        # we dont need checks for mask and image validity, as they done earlier
        asymmetry = mean_asymmetry(mask)
        border = B_compactness(mask)
        color_vector = get_color_vector(img, mask)

        # simplified result dictionary construction using dict unpacking
        # removed intermediate dictionary making and merged directly
        result = {
            'patient_id': patient_id,
            'img_id': img_id,
            'cancer': cancer,
            'border': border,
            'asymmetry': asymmetry
        }
        
        return {**result, **color_vector}
        
    except Exception as e:
        # error handling
        print(f"Error processing {img_id}: {str(e)}")
        return None


def main():
    """
    Main function to orchestrate the feature extraction process.
    
    Major optimizations:
    1. Added parallel processing
    2. Improved file path handling
    3. Optimized data loading and preprocessing
    4. Enhanced memory efficiency
    5. Added progress tracking with tqdm
    """
    # Configuration paths
    image_dir = 'data/imgs/'
    mask_dir = 'data/lesion_masks/'
    metadata_path = 'dataset.csv'
    process_images = "correct_mask_list.csv"
    
    # convert to path objects for more efficient path operations
    # resolve() to make sure we have absolute paths. had a issue with this
    mask_dir = Path(mask_dir).resolve()
    
    
    # removed redundant DataFrame operations and simplified merging
    # read metadata and good pictures in a single step
    # used pandas
    metadata = pd.read_csv(metadata_path)
    good_pics = pd.read_csv(process_images).rename(columns={"Filename": "img_id"})
    
    # Dataframe operations in sequence
    #removed multiple intermediate DFs

    filtered = good_pics.merge(metadata, on='img_id', how='left')
    filtered['cancer'] = filtered['diagnostic'].isin(['BCC', 'MEL', 'SCC'])
    filtered = filtered[['img_id', 'patient_id', 'cancer']]
    
    # dataframe to tuples for more efficency in parallel processing
    # dont need the dictionary conversion
    row_data = list(filtered.itertuples(index=False, name=None))
    
    # Initialize parallel processing
    # Leave one CPU core free to prevent system slowdown
    num_processes = max(cpu_count() - 1, 1)
    
    
    # Set OMP_NUM_THREADS to 1 to prevent nested parallelization conflicts
    # Use partial to create a function with fixed arguments for parallel processing
    # partial is used to bind the image and mask directories to the processing function
    process_func = partial(process_single_image, image_dir=str(image_dir), mask_dir=str(mask_dir))
    
    #process images in parallel using a process pool with progress bar
    total_images = len(row_data)
    print(f"\nProcessing {total_images} images using {num_processes} processes...")
    
    with Pool(processes=num_processes) as pool:
        # Add tqdm progress bar to track parallel processing
        results = list(tqdm(
            pool.imap(process_func, row_data),
            total=total_images,
            desc="Extracting features",
            unit="image"
        ))
    

    valid_results = [r for r in results if r is not None]
    print(f"\nSuccessfully processed {len(valid_results)} out of {total_images} images")
    
    features_df = pd.DataFrame(valid_results)
    
    # Save results using pathlib for better path handling
    output_path = Path('result/enhanced_feature_dataset_2.csv').resolve()
    features_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
