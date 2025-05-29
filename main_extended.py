import pandas as pd
from util.features_extract import feature_extraction
from util.classifier_final import prediction_evaluation, model_training
from pathlib import Path


def main_extended(image_dir, mask_dir, metadata_path, process_images):

    #Loading data with error handling
    try:
        metadata = pd.read_csv(metadata_path)
        good_pics = pd.read_csv(process_images).rename(columns={"Filename": "img_id"})
    except FileNotFoundError as e:
        print(f"Error loading data files: {str(e)}")
        exit(1)

    #Features extraction
    data = feature_extraction(metadata, str(image_dir), str(mask_dir), good_pics, filter=True)
    print(f'Extracted features csv saved to {'../result/feature_dataset.csv'}')

    #Selecting features for baseline model
    features = ['border', 'asymmetry', 'mean_H', 'std_H', 'mean_S', 'std_S', 'mean_V', 'std_V']

    #Training decsion tree model using selected features
    pipe = model_training(data, features)

    #Predicting labels using trained model
    prediction_evaluation(data, features, pipe)

    #Result overview directory
    print(f'Results saved to {'../result/'}')

if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    image_dir = base_dir / 'data' / 'skin_images' / 'original'
    mask_dir = base_dir / 'data' / 'lesion_masks'
    metadata_path = base_dir / 'dataset.csv'
    process_images = base_dir / 'correct_mask_list.csv'

    main_baseline(image_dir, mask_dir, metadata_path, process_images)