import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
from codecarbon import EmissionsTracker
import time
from datetime import datetime
from util.inpaint_util import removeHair
from util.img_util import readImageFile
from util.img_util import readImageFile, saveImageFile
from util.feature_extract import (measure_pigment_network, measure_blue_veil, 
                                     measure_vascular, measure_globules, measure_streaks,
                                     measure_irregular_pigmentation, measure_regression,
                                     get_asymmetry, get_compactness, convexity_score)


def process_single_image(image_path, mask_path, output_dir):
  #Process one image with its mask and extract features.
    
    try:
        # Read image using img_util
        image_rgb, img_gray = readImageFile(image_path)
        if image_rgb is None or img_gray is None:
            return None, None
            
        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None, None
        
        # Hair removal (convert to BGR for removeHair function)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        _, _, image_no_hair = removeHair(image_bgr, img_gray, kernel_size=25, threshold=10, radius=3)
        
        # Create RGBA image with transparency
        rgba = cv2.cvtColor(image_no_hair, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask
        
        # Save masked image
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"masked_{filename}")
        cv2.imwrite(output_path, rgba, [cv2.IMWRITE_PNG_COMPRESSION, 5])
        
        # Extract features (convert back to RGB for feature extraction)
        image_no_hair_rgb = cv2.cvtColor(image_no_hair, cv2.COLOR_BGR2RGB)
        features = {
            'image_name': filename,
            'pigment_network': float(measure_pigment_network(image_no_hair_rgb)),
            'blue_veil': float(measure_blue_veil(image_no_hair_rgb)),
            'vascular': float(measure_vascular(image_no_hair_rgb)),
            'globules': float(measure_globules(image_no_hair_rgb)),
            'streaks': float(measure_streaks(image_no_hair_rgb)),
            'irregular_pigmentation': float(measure_irregular_pigmentation(image_no_hair_rgb)),
            'regression': float(measure_regression(image_no_hair_rgb)),
            'asymmetry': float(get_asymmetry(mask)),
            'compactness': float(get_compactness(mask)),
            'convexity': float(convexity_score(mask))
        }
        
        return filename, features
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, None
    

def process_image_dataset(input_dir, mask_dir, output_dir, n_threads=8):
   #Parrarel processing and feature extract 

    # output dir sanity check
    os.makedirs(output_dir, exist_ok=True)


    # Start carbon tracking
    tracker = EmissionsTracker(project_name="Masking and feature extraction",
                              output_dir=output_dir,
                              log_level='warning')
    tracker.start()
    start_time = time.time()
    print("\033[32mCarbon Tracker Strated... \033", end='\n')
    
    
    # Get list of images
    print("\033[33mScanning input directories... \033", end='\n')
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png') and not f.endswith('_mask.png')]
    total_images = len(image_files)
    
    print(f"\033[35mFound {total_images} images to process\032[0m", end='\n')
    print("\033[37m\n \032[0m")
    
    
    # Name maching for image and mask
    image_paths = [os.path.join(input_dir, f) for f in image_files]
    mask_paths = [os.path.join(mask_dir, f.replace('.png', '_mask.png')) for f in image_files]
    
    features_dict = {}
    successful = 0
    failed = 0
    
    # Main progress bar
    pbar = tqdm(total=total_images, desc="Processing Images", 
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        
        
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        future_to_path = {
            executor.submit(process_single_image, img_path, mask_path, output_dir): img_path
            for img_path, mask_path in zip(image_paths, mask_paths)
        }
        
        for future in as_completed(future_to_path):
            img_path = future_to_path[future]
            try:
                filename, features = future.result()
                if features is not None:
                    features_dict[filename] = features
                    successful += 1
                else:
                    failed += 1
                pbar.set_postfix({"Success": successful, "Failed": failed})
                pbar.update(1)
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                failed += 1
                pbar.update(1)
    
    pbar.close()
    
    # Pndas DF
    df = pd.DataFrame.from_dict(features_dict, orient='index')
    
    # Save features
    df.to_csv(os.path.join(output_dir, 'extracted_features.csv'), index=False)
    
    
    
    # Carbon tracking results
    emissions = tracker.stop()
    end_time = time.time()
    process_time = end_time - start_time
    print("\033[32mCarbon Tracker Killed... \032[0m", end='\n')

    try:
        energy_consumed = tracker.final_energy_consumed
    except AttributeError:
        try:
            energy_consumed = tracker.get_energy_consumed()
        except AttributeError:
            energy_consumed = 0
            print("Warning: Could not retrieve energy consumption data")

    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total images processed: {total_images}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Processing time: {process_time:.2f} seconds")
    print(f"Processing speed: {total_images/process_time:.2f} images/second")
    print("\n" + "="*50)
    print("ENVIRONMENTAL IMPACT")
    print("="*50)
    print(f"Carbon emissions: {emissions:.6f} kg CO2")
    print(f"Energy consumed: {energy_consumed:.2f} kWh")
    print(f"Carbon intensity: {emissions/total_images*1000:.2f} g CO2/image")
    print("="*50)
    
    # Save carbon report
    log_dir = os.path.join(os.path.dirname(os.path.dirname(output_dir)), 'energy_data_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # filenames wit timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f'environmental_report_{timestamp}.txt'
    report_path = os.path.join(log_dir, report_filename)
    
    with open(report_path, 'w') as f:
        f.write(f"Environmental Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Environmental Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n")
        f.write(f"Total images processed: {total_images}\n")
        f.write(f"Carbon emissions: {emissions:.6f} kg CO2\n")
        f.write(f"Energy consumed: {energy_consumed:.2f} kWh\n")
        f.write(f"Carbon intensity: {emissions/total_images*1000:.2f} g CO2/image\n")
        f.write("="*50 + "\n")   

    
    return df

if __name__ == "__main__":

    INPUT_DIR = "data\skin_images"
    MASK_DIR = "data\lesion_masks"
    OUTPUT_DIR = "result\masked_out_images"
    
    features_df = process_image_dataset(INPUT_DIR, MASK_DIR, OUTPUT_DIR)
    print(f"\033[35mProcessed {len(features_df)} images\033")
    #test