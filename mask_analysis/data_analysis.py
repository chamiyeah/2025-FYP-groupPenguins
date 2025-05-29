import os
import pandas as pd
import csv

# File and folder paths
dataset_csv_path = "dataset.csv"
mask_list_csv_path = "mask_making/final_list.csv"

skin_images_folder = "data/skin_images"
all_masks_folder = "data/masks_total"
manual_masks_folder = "data/new_masks/masks"

# Load CSVs
dataset_csv = pd.read_csv(dataset_csv_path)
mask_list_csv = pd.read_csv(mask_list_csv_path)

# List files from folders
skin_image_files = [f for f in os.listdir(skin_images_folder) if f.lower().endswith(".png")]
all_mask_files = [f for f in os.listdir(all_masks_folder) if f.lower().endswith(".png")]
manual_mask_files = [f for f in os.listdir(manual_masks_folder) if f.lower().endswith(".png")]

#1 total images in skin_images
total_skin_images = len(skin_image_files)

#total images in skin_images with a BCC/SCC/MEL diagnosis in dataset.csv
cancer_types = ["BCC", "SCC", "MEL"]
total_cancer_in_dataset = dataset_csv[dataset_csv["diagnostic"].isin(cancer_types)]["img_id"].isin(skin_image_files).sum()

#images not diagnosed with cancer (non-BCC/SCC/MEL) in dataset.csv
num_non_cancer_images = dataset_csv[~dataset_csv["diagnostic"].isin(cancer_types)]["img_id"].isin(skin_image_files).sum()

#total images in masks_total
total_masks = len(all_mask_files)

# total correctly made masks (Correct_Masks) -- from correct mask list CSV
correct_masks_csv = 'mask_making/correct_mask_list.csv'
with open(correct_masks_csv, newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip header if present
    total_correct_masks = sum(1 for row in reader if row)

# total badly made masks (Badly_Made_Masks)
total_badly_made_masks = total_masks - total_correct_masks

#total images in manually created masks (new_masks)
manual_masks_folder = "data/new_masks/masks"
num_manual_masks = len([f for f in os.listdir(manual_masks_folder) if f.lower().endswith('.png')])

#total used mask which are the correct masks + new_masks
num_used_masks = total_correct_masks + num_manual_masks

# Get the list of used images from the final list CSV, normalize to lowercase and .png
used_images = set(mask_list_csv['Filename'].str.strip().str.lower())
# Normalize img_id in dataset_csv
img_id_norm = dataset_csv['img_id'].astype(str).str.strip().str.lower()
if not img_id_norm.str.endswith('.png').all():
    img_id_norm = img_id_norm.apply(lambda x: x if x.endswith('.png') else x + '.png')
dataset_csv['img_id_norm'] = img_id_norm

# Count how many used images are cancer and how many are not (using normalized img_id)
used_cancer = dataset_csv[(dataset_csv['img_id_norm'].isin(used_images)) & (dataset_csv['diagnostic'].isin(cancer_types))]
used_non_cancer = dataset_csv[(dataset_csv['img_id_norm'].isin(used_images)) & (~dataset_csv['diagnostic'].isin(cancer_types))]

# Final summary
print("\n📊 Project Statistics Summary:")
print(f"Total images in dataset (skin_images): {total_skin_images}")
print(f"Images diagnosed with BCC/SCC/MEL: {total_cancer_in_dataset}")
print(f"Images NOT diagnosed with cancer (non-BCC/SCC/MEL): {num_non_cancer_images} \n")

print(f"Total cut outs: {total_masks}")

print(f"cut out that are CANCER: {len(used_cancer)}")
print(f"cut out that are NOT cancer: {len(used_non_cancer)}")
print(f"Manually created masks (new_masks): {num_manual_masks}")
print(f"Total badly made masks (Badly_Made_Masks): {total_badly_made_masks}")
print(f"Total correctly made masks (Correct_Masks): {total_correct_masks}")
print(f"Total used masks (correct + manual): {num_used_masks}")
print(f"Images without a corresponding cut out: {total_skin_images - num_used_masks}")
