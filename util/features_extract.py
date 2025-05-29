import os
os.environ["OMP_NUM_THREADS"] = "1"
import cv2
import numpy as np
import pandas as pd
from util.feature_A import mean_asymmetry
from util.feature_B import B_compactness
from util.feature_C import get_color_vector
from util.image_util import enhance_color_hsv_clahe

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

    if filter:
        #merge to retain only good pictures
        filtered = good_pics.rename(columns={"Filename": "img_id"}).merge(metadata, on='img_id', how='left')
    else:
        filtered = metadata.copy()

    #binary cancer label
    filtered['cancer'] = filtered['diagnostic'].isin(['BCC', 'MEL', 'SCC'])

    filtered = filtered[['img_id', 'patient_id', 'cancer']]

    rows = []
    counter = 0

    #image loader and feature extraction
    for _, row in filtered.iterrows():
        counter += 1
        print(counter)

        img_id = row['img_id']
        patient_id = row['patient_id']
        cancer = row['cancer']

        img_path = os.path.join(image_dir, img_id)
        mask_filename = img_id.replace('.png', '_mask.png')
        mask_path = os.path.join(mask_dir, mask_filename)

        try: #quick check if image and mask exist and convert them to RGB
            img = cv2.imread(img_path)
            if img is None:
                print(f"img None {img_id}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = enhance_color_hsv_clahe(img)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"mask None {img_id}")
                continue

            #feature extraction
            asymmetry = mean_asymmetry(mask)
            border = B_compactness(mask)
            color_vector = get_color_vector(img, mask)  # dict with 6 color features
            #space for open question!!!!!!!!!!!!!!!!

            result = {
                'patient_id': patient_id,
                'img_id': img_id,
                'cancer': cancer,
                'border': border,
                'asymmetry': asymmetry,
                **color_vector
            }

            rows.append(result)

        except Exception as e:
            print(f"Failed on {img_id}: {e}")
            continue

    features_df = pd.DataFrame(rows)
    features_df.to_csv(result_dir / "feature_dataset.csv", index=False)
    return features_df

