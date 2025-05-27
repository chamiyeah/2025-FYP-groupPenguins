import cv2
import numpy as np
import pandas as pd
import os
from feature_A import mean_asymmetry
from feature_B import B_compactness
from feature_C import get_color_vector


# Configuration
image_dir = '../data/imgs_part_1/'
mask_dir = '../data/lesion_masks/'
metadata_path = '../dataset.csv'


# Load main dataframe
metadata = pd.read_csv(metadata_path)


# Create 'cancer' column: True if diagnosis is SCC, BCC, or MEL
metadata['cancer'] = metadata['diagnostic'].isin(['BCC', 'MEL', 'SCC'])


# Preload existing filenames for fast lookup
available_images = set(os.listdir(image_dir))
available_masks = set(os.listdir(mask_dir))


# Initialize list for all feature rows
rows = []
counter = 0

# Iterate and process each sample
for _, row in metadata.iterrows():
    counter += 1
    print(counter)
    img_id = row['img_id']
    patient_id = row['patient_id']
    cancer = row['cancer']

    img_path = os.path.join(image_dir, img_id)
    mask_filename = img_id.replace('.png', '_mask.png')
    mask_path = os.path.join(mask_dir, mask_filename)

    if os.path.basename(img_path) not in available_images or os.path.basename(mask_path) not in available_masks:
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue

    # THIS MIGHT BE REMOVED LATER ON!!!  (we ignore lesions that are multiple and small because the asymmetry border and color)  
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        continue

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

    color_labels = ['White', 'Black', 'Red', 'Light-brown', 'Dark-brown', 'Blue-gray']
    for i, label in enumerate(color_labels):
        result[f'{label}'] = color_vector[i]

    rows.append(result)

# Final DataFrame
features_df = pd.DataFrame(rows)

# Store result
features_df.to_csv('../data/feature_dataset.csv', index=False)