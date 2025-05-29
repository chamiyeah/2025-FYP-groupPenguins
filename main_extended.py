import pandas as pd
from util.classifier_final import prediction_evaluation, model_training
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import cv2
from util.features_extract_extended import feature_extraction_extended
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score,
    precision_score, f1_score, ConfusionMatrixDisplay
)

#Read before running: this file contains 3 scripts, chose one accordingly

##########################################################################################################################################
#use when csv of extracted features is not provided and we want to extract features, train the model and evaluate its performance
def main_extended(image_dir, mask_dir, metadata_path, process_images, result_dir):
    """
    Works the same as a baseline model just with corrected feature extraction function that extracts hair feature
    before hair removal is applied and later extracts all the baseline features.

    Input:
    ----------
        image_dir : str or pathlib.Path
            Directory containing the skin lesion images
        mask_dir : str or pathlib.Path
            Directory containing lesion masks
        metadata_path : str or pathlib.Path
            Path to the CSV file containing diagnostic labels
        process_images : str or pathlib.Path
            Path to the CSV file listing the image IDs considered valid for processing (one lesion per image and visually distinct lesions)
        result_dir : str or pathlib.Path
            Path to directory where all the results are saved in

    Returns:
    -------
        None
            Prints metrics of the extended model on testing data and saves prediction csv to results directory + confusion matrix image
    """
    # Convert all paths to Path objects
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    metadata_path = Path(metadata_path)
    process_images = Path(process_images)
    result_dir = Path(result_dir)

    try:
        metadata = pd.read_csv(str(metadata_path))
        good_pics = pd.read_csv(str(process_images)).rename(columns={"Filename": "img_id"})
    except FileNotFoundError as e:
        print(f"Error loading data files: {str(e)}")
        exit(1)

    print('Starting feature extraction:')
    #Features extraction
    data = feature_extraction_extended(metadata, str(image_dir), str(mask_dir), good_pics, str(result_dir), filter=True)
    print(f"Extracted features csv saved to {result_dir / 'feature_dataset.csv'}")

    #Selecting features for extended model (added hair coverage feature and ABC features are now evealuated after hair removal is done)
    features = ['hair_coverage','border', 'asymmetry', 'mean_H', 'std_H', 'mean_S', 'std_S', 'mean_V', 'std_V', 'color_entropy', 'melanoma_colors']

    #Training decsion tree model using selected features
    pipe = model_training(data, features)

    #Predicting labels and evaluating on test set our trained model pipe
    prediction_evaluation(data, features, pipe, result_dir, name1='confusion_matrix_extended.png', name2='result_extended.csv')

    #Result overview directory
    print(f'Results saved to result directory')

if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    image_dir = base_dir / 'data' / 'skin_images' / 'original'
    mask_dir = base_dir / 'data' / 'lesion_masks'
    metadata_path = base_dir / 'dataset.csv'
    process_images = base_dir / 'correct_mask_list.csv'
    result_dir = base_dir / "result"

    main_extended(image_dir, mask_dir, metadata_path, process_images, result_dir)

################################################################################################################
# #when csv is provided, so we do just training and saving trained model + returning evaluation of the trained model

# def main_extended_with_data(feature_path, result_dir, model_path=None):

#     result_dir = Path(result_dir)
#     feature_path = Path(feature_path)

#     try:
#         data = pd.read_csv(feature_path)
#     except FileNotFoundError as e:
#         print(f"Error loading extracted features CSV: {str(e)}")
#         return

#     print(f"Using extracted features from: {feature_path.name}")

#     features = ['hair_coverage','border', 'asymmetry', 'mean_H', 'std_H', 'mean_S', 'std_S', 'mean_V', 'std_V', 'color_entropy', 'melanoma_colors']

#     pipe = model_training(data, features)

#     #save model if model_path is specified
#     if model_path:
#         model_path = Path(model_path)
#         joblib.dump(pipe, model_path)
#         print(f"Trained model saved to: {model_path}")

#     prediction_evaluation(data, features, pipe, result_dir, name1='confusion_matrix_extended.png', name2='result_extended.csv')

#     print(f"Results saved to: {result_dir}")


# if __name__ == "__main__":
#     base_dir = Path(__file__).parent.resolve()
#     result_dir = base_dir / "result"
#     feature_path = result_dir / "feature_dataset_extended.csv" 
#     model_path = result_dir / "trained_DT_extended.joblib"  

#     main_extended_with_data(feature_path, result_dir, model_path=model_path)


# #########################################################################################################################
# #evaluating on unseen data, the trained model is saved in results directory
# #when ran again on all data that it was trained on and tested on the results will not be perfect because function 
# #is not filtering good images and it havent seen test data, so it works only on dataset that has: good masks and one lesion per image

# def evaluate_unseen_extended(image_dir, mask_dir, metadata_path, model_path, result_dir):

#     image_dir = Path(image_dir)
#     mask_dir = Path(mask_dir)
#     metadata_path = Path(metadata_path)
#     model_path = Path(model_path)
#     result_dir = Path(result_dir)
#     result_dir.mkdir(exist_ok=True, parents=True)

#     try:
#         metadata = pd.read_csv(metadata_path)
#     except FileNotFoundError as e:
#         print(f"Metadata file not found: {e}")
#         return
 
#     #filter metadata to only rows that have masks present
#     mask_filenames = set(f.name for f in mask_dir.glob("*_mask.png"))
#     metadata['has_mask'] = metadata['img_id'].apply(
#         lambda x: x.replace('.png', '_mask.png') in mask_filenames
#     )
#     filtered_metadata = metadata[metadata['has_mask']].drop(columns="has_mask")
#     print(f"Total images in metadata: {len(metadata)}")
#     print(f"Images with matching masks: {len(filtered_metadata)}")

#     if len(filtered_metadata) == 0:
#         print("No valid images with masks found. Aborting.")
#         return

#     print("Extracting features from images with masks...")
#     extracted = feature_extraction_extended(
#         filtered_metadata, str(image_dir), str(mask_dir),
#         good_pics=None, result_dir=result_dir, filter=False
#     )

#     print("Features extracted and saved.")
#     features_csv_path = result_dir / "feature_dataset.csv"

#     #load trained model
#     try:
#         model = joblib.load(model_path)
#     except FileNotFoundError as e:
#         print(f"Trained model not found: {e}")
#         return

#     extracted['cancer'] = extracted['cancer'].astype(int)
#     features = ['hair_coverage','border', 'asymmetry', 'mean_H', 'std_H', 'mean_S', 'std_S', 'mean_V', 'std_V', 'color_entropy', 'melanoma_colors']
#     X_test = extracted[features]
#     y_test = extracted['cancer']

#     #comback if there is time! 
#     #could not reuse prediction_evaluation function since it lacks logic of inputs, i made it too specific for training
#     y_pred = model.predict(X_test)
#     y_prob = model.predict_proba(X_test)[:, 1]

#     acc = accuracy_score(y_test, y_pred)
#     rec = recall_score(y_test, y_pred)
#     auc = roc_auc_score(y_test, y_prob)
#     prec = precision_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)

#     print("\nEvaluation on Unseen Dataset:")
#     print(f"Accuracy:  {acc:.4f}")
#     print(f"Recall:    {rec:.4f}")
#     print(f"AUC:       {auc:.4f}")
#     print(f"Precision: {prec:.4f}")
#     print(f"F1 Score:  {f1:.4f}")

#     ConfusionMatrixDisplay.from_predictions(
#         y_test, y_pred,
#         display_labels=["Benign", "Cancer"],
#         cmap="Oranges",
#         values_format="d"
#     )
#     plt.title("Confusion Matrix on Unseen Set")
#     plt.grid(False)
#     plt.savefig(result_dir / "confusion_matrix_unseen_extended.png", dpi=300)
#     plt.show()

#     results_df = pd.DataFrame({
#         "img_id": extracted["img_id"],
#         "patient_id": extracted["patient_id"],
#         "label": y_test,
#         "probability": y_prob
#     })
#     results_df.to_csv(result_dir / "results_unseen_test_extended.csv", index=False)
#     print(f"Saved prediction results to {result_dir / 'results_unseen_test_extended.csv'}")


# if __name__ == "__main__":
#     base_dir = Path(__file__).parent.resolve()

#     evaluate_unseen_extended(
#         image_dir=base_dir / "data" / "skin_images" / "original",
#         mask_dir=base_dir / "data" / "lesion_masks",
#         metadata_path=base_dir / "dataset.csv",
#         model_path=base_dir / "result" / "trained_DT_extended.joblib",
#         result_dir=base_dir / "result"
#     )
