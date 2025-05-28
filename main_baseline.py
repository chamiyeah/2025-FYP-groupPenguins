import pandas as pd
from util.features_extract import feature_extraction
from util.classifier_final import prediction_evaluation

image_dir = './data/skin_images/original/'
mask_dir = './data/lesion_masks/'
metadata_path = 'dataset.csv'
process_images = "correct_mask_list.csv"

metadata = pd.read_csv(metadata_path)
good_pics = pd.read_csv(process_images).rename(columns={"Filename": "img_id"})

data = feature_extraction(metadata, image_dir, mask_dir, good_pics, filter=True)

features = ['border', 'asymmetry', 'mean_H', 'std_H', 'mean_H', 'mean_S', 'std_S', 'mean_V', 'std_V']

prediction_evaluation(data, features)