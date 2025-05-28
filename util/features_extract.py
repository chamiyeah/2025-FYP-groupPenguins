import cv2
import numpy as np
import pandas as pd
import os
from feature_A import mean_asymmetry
from feature_B import B_compactness
from feature_C import get_color_vector


# Configuration
image_dir = '../data/imgs/'
mask_dir = '../data/lesion_masks/'
metadata_path = '../dataset.csv'
process_images = "../good_images_list.csv"


# Load both CSVs
metadata = pd.read_csv(metadata_path)
good_pics = pd.read_csv(process_images).rename(columns={"Filename": "img_id"})


# Merge to keep only good pictures (left join keeps order of good_pics)
filtered = good_pics.merge(metadata, on='img_id', how='left')

# Create 'cancer' column: True if diagnosis is SCC, BCC, or MEL
filtered['cancer'] = filtered['diagnostic'].isin(['BCC', 'MEL', 'SCC'])

# Choose the columns were gonna need
filtered = filtered[['img_id', 'patient_id', 'cancer']]

# # Preload existing filenames for fast lookup
# available_images = set(os.listdir(image_dir))
# available_masks = set(os.listdir(mask_dir))


# Initialize list for all feature rows
rows = []
counter = 0

# Iterate and process each sample
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

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"img None {img_id}")
            continue

        # THIS MIGHT BE REMOVED LATER ON!!!  (we ignore lesions that are multiple and small because the asymmetry border and color are not gonna be trustworthy)
        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if len(contours) > 1:
        #     print(f"several lesions, {img_id}")
        #     continue

        # Extract features
        asymmetry = mean_asymmetry(mask)
        border = B_compactness(mask)
        color_vector = get_color_vector(img, mask)  # array of 6 elements

        # Build row
        result = {
            'patient_id': patient_id,
            'img_id': img_id,
            'cancer': cancer,
            'border': border,
            'asymmetry': asymmetry
        }

        final_result = {**result, **color_vector}

        # color_labels = ['White', 'Black', 'Red', 'Light-brown', 'Dark-brown', 'Blue-gray']
        # for i, label in enumerate(color_labels):
        #     result[f'{label}'] = color_vector[i]

        rows.append(final_result)

    except:
        continue

# Final DataFrame
features_df = pd.DataFrame(rows)

# Store result
features_df.to_csv('feature_dataset.csv', index=False)
