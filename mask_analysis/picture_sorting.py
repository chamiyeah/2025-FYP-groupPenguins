import os
import pandas as pd
import shutil

# Sorts the images and masks into files based on the mask analysis file after being filled out
# Images divided into: Filtered images if they are fine, or excluded images if they were deemed unfit for the project
# Masks were divided into: Correct masks if everything is fine, or badly made masks if they need to be redone.

#Requires all the lesion images to be inside the "skin_images" folder in the exisiting folder called "data"

base_path = os.path.dirname(__file__)

csv_path = os.path.join(base_path, "mask_analysis.csv")
images_path = os.path.abspath(os.path.join(base_path, "..", "data", "skin_images"))
masks_path = os.path.abspath(os.path.join(base_path, "..", "masked_out_lesions"))

excluded_path = os.path.abspath(os.path.join(base_path, "..", "data", "Excluded_Images"))
filtered_path = os.path.abspath(os.path.join(base_path, "..", "data", "Filtered_Images"))
masks_good_path = os.path.abspath(os.path.join(base_path, "..", "data", "Correct_Masks"))
masks_bad_path = os.path.abspath(os.path.join(base_path, "..", "data", "Badly_Made_Masks"))

os.makedirs(excluded_path, exist_ok=True)
os.makedirs(filtered_path, exist_ok=True)
os.makedirs(masks_good_path, exist_ok=True)
os.makedirs(masks_bad_path, exist_ok=True)

df = pd.read_csv(csv_path)

for _, row in df.iterrows():
    filename = row["Filename"]
    src = os.path.join(images_path, filename)

    if not os.path.exists(src):
        continue

    if row["Exclude"] == 1:
        dst = os.path.join(excluded_path, filename)
    else:
        dst = os.path.join(filtered_path, filename)

    shutil.copy2(src, dst)

for _, row in df.iterrows():
    filename = row["Filename"]
    name, ext = os.path.splitext(filename)
    mask_name = f"{name}_chopped{ext}"
    src_mask = os.path.join(masks_path, mask_name)

    if not os.path.exists(src_mask):
        continue
    
    if row["Remake mask"] == 1:
        dst_mask = os.path.join(masks_bad_path, mask_name)
    else:
        dst_mask = os.path.join(masks_good_path, mask_name)

    shutil.copy2(src_mask, dst_mask)

