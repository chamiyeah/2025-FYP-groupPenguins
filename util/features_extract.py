import os
os.environ["OMP_NUM_THREADS"] = "1"
import cv2
import numpy as np
import pandas as pd
from feature_A import mean_asymmetry
from feature_B import B_compactness
from feature_C import get_color_vector
from image_preprocessing import enhance_color_hsv_clahe

image_dir = '../data/skin_images/original/'
mask_dir = '../data/lesion_masks/'
metadata_path = '../dataset.csv'
process_images = "../correct_mask_list.csv"

metadata = pd.read_csv(metadata_path)
good_pics = pd.read_csv(process_images).rename(columns={"Filename": "img_id"})


def feature_extraction(metadata, image_dir, mask_dir, good_pics, filter=True):
    """
    Extracts features from lesion images and corresponding masks. 
    If filter=True, uses a list of valid image IDs (good_pics) to filter input.

    Parameters:
        metadata (pd.DataFrame): DataFrame containing metadata for all images.
        image_dir (str): Directory path to input images.
        mask_dir (str): Directory path to segmentation masks.
        good_pics (pd.DataFrame): DataFrame with column 'img_id' listing valid image files.
        filter (bool): If True, filters the data using good_pics; otherwise uses all images in metadata.

    Returns:
        None: Saves extracted features to '../result/feature_dataset.csv'.
    """
    import os
    import cv2
    import pandas as pd

    if filter:
        # Merge to retain only good pictures
        filtered = good_pics.rename(columns={"Filename": "img_id"}).merge(metadata, on='img_id', how='left')
    else:
        filtered = metadata.copy()

    # Add binary cancer label
    filtered['cancer'] = filtered['diagnostic'].isin(['BCC', 'MEL', 'SCC'])

    # Keep only relevant columns
    filtered = filtered[['img_id', 'patient_id', 'cancer']]

    rows = []
    counter = 0

    for _, row in filtered.iterrows():
        counter += 1
        print(counter)

        img_id = row['img_id']
        patient_id = row['patient_id']
        cancer = row['cancer']

        img_path = os.path.join(image_dir, img_id)
        mask_filename = img_id.replace('.png', '_mask.png')
        mask_path = os.path.join(mask_dir, mask_filename)

        try:
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

            # Feature extraction
            asymmetry = mean_asymmetry(mask)
            border = B_compactness(mask)
            color_vector = get_color_vector(img, mask)  # dict with 6 color features

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

    # Save final feature set
    features_df = pd.DataFrame(rows)
    features_df.to_csv('../result/feature_dataset.csv', index=False)
    return features_df

feature_extraction(metadata, image_dir, mask_dir, good_pics, filter=True)
print('yes')

#########################################################################################
# # Configuration
# image_dir = '../data/imgs/'
# mask_dir = '../data/lesion_masks/'
# metadata_path = '../dataset.csv'
# process_images = "../correct_mask_list.csv"

# # Load both CSVs
# metadata = pd.read_csv(metadata_path)
# good_pics = pd.read_csv(process_images).rename(columns={"Filename": "img_id"}

# #merge to keep only good pictures (left join keeps order of good_pics)
# filtered = good_pics.merge(metadata, on='img_id', how='left')

# #create 'cancer' column: True if diagnosis is SCC, BCC, or MEL
# filtered['cancer'] = filtered['diagnostic'].isin(['BCC', 'MEL', 'SCC'])

# # Choose the columns were gonna need
# filtered = filtered[['img_id', 'patient_id', 'cancer']]

# # # Preload existing filenames for fast lookup
# # available_images = set(os.listdir(image_dir))
# # available_masks = set(os.listdir(mask_dir))


# # Initialize list for all feature rows
# rows = []
# counter = 0

# # Iterate and process each sample
# for _, row in filtered.iterrows():
#     counter += 1
#     print(counter)
#     img_id = row['img_id']
#     patient_id = row['patient_id']
#     cancer = row['cancer']

#     img_path = os.path.join(image_dir, img_id)
#     mask_filename = img_id.replace('.png', '_mask.png')
#     mask_path = os.path.join(mask_dir, mask_filename)

#     try:
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"img None {img_id}")
#             continue
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = enhance_color_hsv_clahe(img)

#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             print(f"img None {img_id}")
#             continue

#         # THIS MIGHT BE REMOVED LATER ON!!!  (we ignore lesions that are multiple and small because the asymmetry border and color are not gonna be trustworthy)
#         # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         # if len(contours) > 1:
#         #     print(f"several lesions, {img_id}")
#         #     continue

#         # Extract features
#         asymmetry = mean_asymmetry(mask)
#         border = B_compactness(mask)
#         color_vector = get_color_vector(img, mask)  # array of 6 elements

#         # Build row
#         result = {
#             'patient_id': patient_id,
#             'img_id': img_id,
#             'cancer': cancer,
#             'border': border,
#             'asymmetry': asymmetry
#         }

#         final_result = {**result, **color_vector}

#         # color_labels = ['White', 'Black', 'Red', 'Light-brown', 'Dark-brown', 'Blue-gray']
#         # for i, label in enumerate(color_labels):
#         #     result[f'{label}'] = color_vector[i]

#         rows.append(final_result)

#     except:
#         continue

# # Final DataFrame
# features_df = pd.DataFrame(rows)

# # Store result
# features_df.to_csv('../result/feature_dataset.csv', index=False)
