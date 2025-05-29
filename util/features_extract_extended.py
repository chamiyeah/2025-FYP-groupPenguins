import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from util.inpaint_final import hair_coverage, removeHair
from util.feature_A import mean_asymmetry
from util.feature_B import B_compactness
from util.feature_C import get_color_vector
from util.image_util import enhance_color_hsv_clahe
from util.feature_T import mean_gradient
from tqdm import tqdm

# ANSI color codes for terminal output
YELLOW = '\033[93m'
PURPLE = '\033[95m'
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

def feature_extraction_extended(metadata_path, image_dir, mask_dir, good_pics_path, result_dir, filter=True):
    """
    Extracts features from lesion images and corresponding masks. If filter=True, uses a list of valid image IDs (good_pics) to filter input.
    Includes hair detection (within lesion only) and hair removal before computing lesion features.

    Input:
        metadata_path (str or Path): Path to metadata CSV file
        image_dir (str): Directory path to input images
        mask_dir (str): Directory path to segmentation masks
        good_pics_path (str or Path): Path to CSV file listing valid image IDs (with column 'Filename')
        filter (bool): If True, filters the data using good_pics; otherwise uses all images in metadata
        result_dir (str or pathlib.Path): Path to directory where all the results are saved in

    Returns:
        pd.DataFrame: DataFrame with extracted features (also saved to CSV)
    """

    print(f"\n{YELLOW}Initializing feature extraction process...{RESET}")
    print(f"{YELLOW}Loading metadata and configuration...{RESET}")
    
    metadata = pd.read_csv(metadata_path)
    good_pics = pd.read_csv(good_pics_path)

    if filter:
        print(f"{PURPLE}Filtering images based on quality criteria...{RESET}")
        filtered = good_pics.rename(columns={"Filename": "img_id"}).merge(metadata, on='img_id', how='left')
    else:
        filtered = metadata.copy()

    filtered['cancer'] = filtered['diagnostic'].isin(['BCC', 'MEL', 'SCC'])
    filtered = filtered[['img_id', 'patient_id', 'cancer']]
    
    total_images = len(filtered)
    print(f"{GREEN}Found {total_images} images to process{RESET}")

    rows = []
    counter = 0

    #data loader and feature extraction
    print(f"{PURPLE}Starting feature extraction for each image...{RESET}")
    for _, row in tqdm(filtered.iterrows(), total=total_images, desc=f"{PURPLE}Processing images{RESET}", ncols=100):
        counter += 1

        img_id = row['img_id']
        patient_id = row['patient_id']
        cancer = row['cancer']

        img_path = os.path.join(image_dir, img_id)
        mask_filename = img_id.replace('.png', '_mask.png')
        mask_path = os.path.join(mask_dir, mask_filename)

        try:#validity check for image and mask existence
            img = cv2.imread(img_path)
            if img is None:
                print(f"{RED}Error: Image not found or corrupted - {img_id}{RESET}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"{RED}Error: Mask not found or corrupted - {img_id}{RESET}")
                continue

            coverage = hair_coverage(img_gray, mask)
            img_clean = removeHair(img_rgb, img_gray)
            img_clean = enhance_color_hsv_clahe(img_clean)

            asymmetry = mean_asymmetry(mask)
            border = B_compactness(mask)
            color_vector = get_color_vector(img_clean, mask)
            texture = mean_gradient(img, mask)

            result = {
                'patient_id': patient_id,
                'img_id': img_id,
                'cancer': cancer,
                'hair_coverage': coverage,
                'border': border,
                'asymmetry': asymmetry,
                'texture': texture,
                **color_vector
            }

            rows.append(result)

        except Exception as e:
            print(f"{RED}Failed processing {img_id}: {e}{RESET}")
            continue

    print(f"\n{PURPLE}Creating features DataFrame...{RESET}")
    features_df = pd.DataFrame(rows)
    features_df.to_csv(Path(result_dir) / "feature_dataset_extended.csv", index=False)
    print(f"{GREEN}Successfully processed {len(features_df)} images{RESET}")
    print(f"{GREEN}Features saved to: {Path(result_dir) / 'feature_dataset_extended.csv'}{RESET}")
    return features_df


if __name__ == "__main__":
    # base_dir = Path(__file__).parent.resolve()
    metadata_path = '../dataset.csv'
    image_dir =  '../data/skin_images/original/'
    mask_dir = '../data/lesion_masks/'
    process_images = '../correct_mask_list.csv'
    result_dir = "../result/"

    feature_extraction_extended(metadata_path, image_dir, mask_dir, process_images, result_dir)