import pandas as pd
from util.features_extract import feature_extraction
from util.classifier_final import prediction_evaluation
from pathlib import Path

# Use proper path resolution
base_dir = Path(__file__).parent.resolve()
image_dir = base_dir / 'data' / 'imgs'
mask_dir = base_dir / 'data' / 'lesion_masks'
metadata_path = base_dir / 'dataset.csv'
process_images = base_dir / 'correct_mask_list.csv'

# Load data with error handling
try:
    metadata = pd.read_csv(metadata_path)
    good_pics = pd.read_csv(process_images).rename(columns={"Filename": "img_id"})
except FileNotFoundError as e:
    print(f"Error loading data files: {str(e)}")
    exit(1)

# Extract features
data = feature_extraction(metadata, str(image_dir), str(mask_dir), good_pics, filter=True)

# Correct feature list (removed duplicate mean_H)
features = ['border', 'asymmetry', 'mean_H', 'std_H', 'mean_S', 'std_S', 'mean_V', 'std_V']

# Run prediction evaluation
prediction_evaluation(data, features)