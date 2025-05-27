import os
import pandas as pd
import shutil

# Define base paths
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, "mask_analysis.csv")
images_path = os.path.abspath(os.path.join(base_path, "..", "data", "skin_images"))
masks_path = os.path.abspath(os.path.join(base_path, "..", "masked_out_lesions"))
review_path = os.path.abspath(os.path.join(base_path, "..", "data", "review"))
review_csv_path = os.path.join(base_path, "review_list.csv")

# # Create the 'review' folder if it doesn't exist
# os.makedirs(review_path, exist_ok=True)

# # Read the CSV file
# df = pd.read_csv(csv_path)

# # Store filenames for the review list
# review_filenames = []

# for _, row in df.iterrows():
#     if row["Exclude"] == 1 or row["Remake mask"] == 1:
#         filename = row["Filename"]
#         name, ext = os.path.splitext(filename)

#         img_src = os.path.join(images_path, filename)
#         mask_src = os.path.join(masks_path, f"{name}_chopped{ext}")
#         img_dst = os.path.join(review_path, filename)
#         mask_dst = os.path.join(review_path, f"{name}_chopped{ext}")

#         # Force overwrite image if it exists
#         if os.path.exists(img_src):
#             shutil.copy2(img_src, img_dst)
#             print(f"Image overwritten: {filename}")
#             review_filenames.append(filename)
#         else:
#             print(f"Image missing, skipped: {filename}")
#             continue  # skip mask copy if image is missing

#         # Force overwrite mask if it exists
#         if os.path.exists(mask_src):
#             shutil.copy2(mask_src, mask_dst)
#             print(f"Mask overwritten: {name}_chopped{ext}")
#         else:
#             print(f"Mask missing, skipped: {name}_chopped{ext}")

# # Write the review list CSV
# if review_filenames:
#     review_df = pd.DataFrame(review_filenames, columns=["Filename"])
#     review_df["Review"] = 0
#     review_df.to_csv(review_csv_path, index=False)
#     print(f"\nReview list CSV created at: {review_csv_path}")
# else:
#     print("\nNo valid files were added. Review list CSV not created.")

print("\nSorting files based on review_list.csv...")

# Load the updated review CSV
df = pd.read_csv(review_csv_path)

# Define destination folders
excluded_images_path = os.path.abspath(os.path.join(base_path, "..", "data", "Excluded_Images"))
filtered_images_path = os.path.abspath(os.path.join(base_path, "..", "data", "Filtered_Images"))
remask_images_path = os.path.abspath(os.path.join(base_path, "..", "data", "To_Remask_Images"))
bad_masks_path = os.path.abspath(os.path.join(base_path, "..", "data", "Badly_Made_Masks"))
correct_masks_path = os.path.abspath(os.path.join(base_path, "..", "data", "Correct_Masks"))

# Create folders if they don't exist
os.makedirs(excluded_images_path, exist_ok=True)
os.makedirs(filtered_images_path, exist_ok=True)
os.makedirs(remask_images_path, exist_ok=True)
os.makedirs(bad_masks_path, exist_ok=True)
os.makedirs(correct_masks_path, exist_ok=True)

# Counters
missing_images = 0
missing_masks = 0

for _, row in df.iterrows():
    filename = row["Filename"]
    review_code = row["Review"]
    name, ext = os.path.splitext(filename)

    img_src = os.path.join(images_path, filename)
    mask_src = os.path.join(masks_path, f"{name}_chopped{ext}")

    if not os.path.exists(img_src):
        missing_images += 1
        continue

    # === IMAGE DESTINATION ===
    if review_code == 0:
        shutil.copy2(img_src, os.path.join(excluded_images_path, filename))
    elif review_code == 1:
        shutil.copy2(img_src, os.path.join(remask_images_path, filename))
    elif review_code == 2:
        shutil.copy2(img_src, os.path.join(filtered_images_path, filename))

    # === MASK DESTINATION ===
    if review_code in [0, 1]:
        dst = os.path.join(bad_masks_path, f"{name}_chopped{ext}")
    elif review_code == 2:
        dst = os.path.join(correct_masks_path, f"{name}_chopped{ext}")
    else:
        dst = None

    if dst:
        if os.path.exists(mask_src):
            shutil.copy2(mask_src, dst)
        else:
            missing_masks += 1

# Final summary
print(f"Done. {missing_images} missing images, {missing_masks} missing masks.")
