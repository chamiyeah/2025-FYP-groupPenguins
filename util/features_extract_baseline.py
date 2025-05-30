import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))
sys.path.append(str(Path(__file__).parent.parent.resolve()))

import os
os.environ["OMP_NUM_THREADS"] = "1"
import cv2
import numpy as np
import pandas as pd
from util.feature_A import mean_asymmetry
from util.feature_B import B_compactness
from util.feature_C import get_color_vector
from util.image_util import enhance_color_hsv_clahe
from util.feature_T import mean_gradient
import concurrent.futures
from tqdm import tqdm

def feature_extraction(metadata, image_dir, mask_dir, good_pics, result_dir, filter=True):
    """
    Extracts features from lesion images and corresponding masks. 
    If filter=True, uses a list of valid image IDs (good_pics) to filter input.

    Input:
        metadata (pd.DataFrame): DataFrame labels
        image_dir (str): original images directory
        mask_dir (str): masks directory
        good_pics (pd.DataFrame): dataframe with column 'img_id' listing valid image files
        filter (bool): if true, filters the data using good_pics (valid pictures); otherwise uses all images in metadata
        result_dir (str or pathlib.Path) : path to directory where all the results are saved in
    Returns:
        Features dataframe and saves it to '../result/feature_dataset.csv'
    """

    YELLOW = '\033[93m'
    PURPLE = '\033[95m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

    if filter:
        #merge to retain only good pictures
        filtered = good_pics.rename(columns={"Filename": "img_id"}).merge(metadata, on='img_id', how='left')
    else:
        filtered = metadata.copy()

    #binary cancer label
    filtered['cancer'] = filtered['diagnostic'].isin(['BCC', 'MEL', 'SCC'])

    filtered = filtered[['img_id', 'patient_id', 'cancer']]

    total_images = len(filtered)
    print(f"{YELLOW}Starting feature extraction for {total_images} images...{RESET}")

    def process_single_image(row):
        img_id = row['img_id']
        patient_id = row['patient_id']
        cancer = row['cancer']
        img_path = os.path.join(image_dir, img_id)
        mask_filename = img_id.replace('.png', '_mask.png')
        mask_path = os.path.join(mask_dir, mask_filename)
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"{RED}img None {img_id}{RESET}")
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = enhance_color_hsv_clahe(img)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"{RED}mask None {img_id}{RESET}")
                return None
            #feature extraction
            asymmetry = mean_asymmetry(mask)
            border = B_compactness(mask)
            color_vector = get_color_vector(img, mask) 
            texture = mean_gradient(img, mask)

            result = {
                'patient_id': patient_id,
                'img_id': img_id,
                'cancer': cancer,
                'border': border,
                'asymmetry': asymmetry,
                'texture': texture,
                **color_vector
            }

            return result
        except Exception as e:
            print(f"{RED}Failed on {img_id}: {e}{RESET}")
            return None

    rows = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_image, [row for _, row in filtered.iterrows()]), total=total_images, desc=f"{PURPLE}Processing images{RESET}", ncols=100))
    rows = [r for r in results if r is not None]

    features_df = pd.DataFrame(rows)
    features_df.to_csv(result_dir / "feature_dataset.csv", index=False)
    print(f"{GREEN}Feature extraction complete. Saved to {result_dir / 'feature_dataset.csv'}{RESET}")
    return features_df

