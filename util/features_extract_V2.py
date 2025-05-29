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
from feature_C import get_color_vector, melanoma_color_labels
from image_util import enhance_color_hsv_clahe


def process_single_image(row_data, image_dir, mask_dir):
    """
    Process a single image to extract features. #This has to be done if we want to use parallel processing effectively.
    
    Optimizations :
    1. function signature will accept tuple of data instead of individual parameters
    2. used more efficient path joining with os.path.join
    3. easier return structure for parallel processing
    """
    img_id = None  # Initialize img_id outside try block
    try:
        # Unpack the row data tuple
        img_id, patient_id, cancer = row_data
        
        # Ensure we have absolute paths
        img_path = os.path.join(image_dir, img_id)
        mask_filename = img_id.replace('.png', '_mask.png')
        mask_path = os.path.join(mask_dir, mask_filename)

        # Verify file existence before reading
        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
            return None
        if not os.path.exists(mask_path):
            print(f"Mask file not found: {mask_path}")
            return None

        # Read image with explicit flags and error checking
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to read image: {img_id}")
            return None
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = enhance_color_hsv_clahe(img)
        
        # Read mask with explicit flags
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to read mask: {img_id}")
            return None

        # Extract features with additional error checking
        try:
            asymmetry = mean_asymmetry(mask)
            border = B_compactness(mask)
            color_vector = get_color_vector(img, mask)

            # Skip processing if any of the features failed
            if any(x is None for x in [asymmetry, border, color_vector]):
                print(f"Feature extraction returned None for {img_id}")
                return None

            # More comprehensive color vector validation
            required_keys = [
                'mean_H', 'std_H', 'mean_S', 'std_S', 'mean_V', 'std_V',  # HSV statistics
                'color_entropy', 'dominant_colors'  # Additional features
            ] + melanoma_color_labels  # Color class proportions
            
            if not isinstance(color_vector, dict):
                print(f"Color vector is not a dictionary for {img_id}")
                return None
                
            if not all(k in color_vector for k in required_keys):
                missing_keys = [k for k in required_keys if k not in color_vector]
                print(f"Missing keys in color vector for {img_id}: {missing_keys}")
                return None
                
            # Simplified type checking that handles both scalar and array types
            if not all(np.isscalar(color_vector[k]) for k in required_keys):
                invalid_keys = [k for k in required_keys if not np.isscalar(color_vector[k])]
                print(f"Invalid value types in color vector for {img_id}: {invalid_keys}")
                return None

        except Exception as feat_error:
            print(f"Feature extraction failed for {img_id}: {str(feat_error)}")
            return None

        # Build result dictionary
        result = {
            'patient_id': patient_id,
            'img_id': img_id,
            'cancer': cancer,
            'border': border,
            'asymmetry': asymmetry
        }
        
        return {**result, **color_vector}
        
    except Exception as e:
        error_msg = f"Error processing {img_id if img_id else 'unknown image'}: {str(e)}"
        print(error_msg)
        return None


def main():
    """
    Main function to orchestrate the feature extraction process.
    """
    # Configuration paths with proper resolution
    base_dir = Path(__file__).parent.parent.resolve()
    image_dir = base_dir / 'data' / 'imgs'
    mask_dir = base_dir / 'data' / 'lesion_masks'
    metadata_path = base_dir / 'dataset.csv'
    process_images = base_dir / 'correct_mask_list.csv'
    
    # Verify directories exist
    if not all(p.exists() for p in [image_dir, mask_dir, metadata_path, process_images]):
        raise FileNotFoundError("Required directories or files not found")
    
    # Load data with error checking
    try:
        metadata = pd.read_csv(metadata_path)
        good_pics = pd.read_csv(process_images).rename(columns={"Filename": "img_id"})
    except Exception as e:
        raise Exception(f"Failed to load data: {str(e)}")
    
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
    process_func = partial(process_single_image, 
                         image_dir=str(image_dir), 
                         mask_dir=str(mask_dir))
    
    # Process images with progress tracking
    total_images = len(row_data)
    print(f"\nProcessing {total_images} images using {num_processes} processes...")
    
    with Pool(processes=num_processes) as pool:

        results = list(tqdm(
            pool.imap(process_func, row_data),

            total=total_images,
            desc="Extracting features",
# Add tqdm progress bar to track parallel processing
            unit="image"
        ))
    
    # Filter and validate results
    valid_results = [r for r in results if r is not None]
    failed_count = total_images - len(valid_results)
    print(f"\nSuccessfully processed {len(valid_results)} out of {total_images} images")
    if failed_count > 0:
        print(f"Failed to process {failed_count} images")
    
    # Create and save DataFrame
    features_df = pd.DataFrame(valid_results)
    output_path = base_dir / 'result' / 'enhanced_feature_dataset.csv'
    features_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
