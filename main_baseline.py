import pandas as pd
from util.features_extract import feature_extraction
from util.classifier_final import prediction_evaluation, model_training
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import cv2
from util.features_extract import feature_extraction
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score,
    precision_score, f1_score, ConfusionMatrixDisplay
)

#file contains 3 scripts, chose one accordingly
#########################################################################################################################
#when csv of extracted features is not provided and we want to train the model and evaluate its performance
# def main_baseline(image_dir, mask_dir, metadata_path, process_images, result_dir):
#     """
#     Baseline model pipeline:
#     - Loads metadata and list of valid images (process_images)
#     - Extracts features from filtered image/mask pairs
#     - Trains a Decision Tree model on selected baseline features
#     - Evaluates the model on a patient-level split
#     - Prints metrics and saves results in result directory

#     Input:
#     ----------
#         image_dir : str or pathlib.Path
#             Directory containing the skin lesion images
#         mask_dir : str or pathlib.Path
#             Directory containing lesion masks
#         metadata_path : str or pathlib.Path
#             Path to the CSV file containing diagnostic labels
#         process_images : str or pathlib.Path
#             Path to the CSV file listing the image IDs considered valid for processing (one lesion per image and visually distinct lesions)
#         result_dir : str or pathlib.Path
#             Path to directory where all the results are saved in

#     Returns:
#     -------
#         None
#             Prints metrics of the baseline model on tetsing data and saves prediction csv to results directory + confusion matrix image
#     """
#     #Loading data with error handling
#     try:
#         metadata = pd.read_csv(metadata_path)
#         good_pics = pd.read_csv(process_images).rename(columns={"Filename": "img_id"})
#     except FileNotFoundError as e:
#         print(f"Error loading data files: {str(e)}")
#         exit(1)

#     print('Starting feature extraction:')
#     #Features extraction
#     data = feature_extraction(metadata, str(image_dir), str(mask_dir), good_pics, result_dir, filter=True)
#     print(f'Extracted features csv saved to {'./result/feature_dataset.csv'}')

#     #Selecting features for baseline model
#     features = ['border', 'asymmetry', 'mean_H', 'std_H', 'mean_S', 'std_S', 'mean_V', 'std_V', 'color_entropy', 'melanoma_colors']

#     #Training decsion tree model using selected features
#     pipe = model_training(data, features)

#     #Predicting labels and evaluating on test set our trained model pipe
#     prediction_evaluation(data, features, pipe, result_dir, name1='confusion_matrix_baseline.png', name2='result_baseline.csv')

#     #Result overview directory
#     print(f'Results saved to result directory')

# if __name__ == "__main__":
#     base_dir = Path(__file__).parent.resolve()
#     image_dir = base_dir / 'data' / 'skin_images' / 'original'
#     mask_dir = base_dir / 'data' / 'lesion_masks'
#     metadata_path = base_dir / 'dataset.csv'
#     process_images = base_dir / 'correct_mask_list.csv'
#     result_dir = base_dir / "result"

#     main_baseline(image_dir, mask_dir, metadata_path, process_images, result_dir)

################################################################################################################
# #when csv is provided, training model and saving trained model + returning evaluation of the trained model

def main_baseline_with_data(feature_path, result_dir, model_path=None):
    """
    Baseline model pipeline (when extracted features DataFrame is already available):
    - Loads the extracted features CSV file from the given path
    - Trains a Decision Tree model on selected baseline features
    - Saves the trained model if a path is provided
    - Evaluates the model on a patient-level split
    - Prints metrics and saves results in the result directory

    Input:
    ----------
        feature_path : str or pathlib.Path
            Path to the extracted features CSV file
        result_dir : str or pathlib.Path
            Path to directory where all the results will be stored
        model_path : str or pathlib.Path, optional
            If provided, saves the trained model to this path

    Returns:
    -------
        None
            Prints metrics of the baseline model on testing data and
            saves prediction CSV and confusion matrix image to result directory
    """
    result_dir = Path(result_dir)
    feature_path = Path(feature_path)

    try:
        data = pd.read_csv(feature_path)
    except FileNotFoundError as e:
        print(f"Error loading extracted features CSV: {str(e)}")
        return

    print(f"Using extracted features from: {feature_path.name}")

    features = ['border', 'asymmetry', 'mean_H', 'std_H', 'mean_S', 'std_S', 'mean_V', 'std_V', 'color_entropy', 'melanoma_colors'] #'color_entropy', 

    pipe = model_training(data, features)

    #save model if model_path is specified
    if model_path:
        model_path = Path(model_path)
        joblib.dump(pipe, model_path)
        print(f"Trained model saved to: {model_path}")

    prediction_evaluation(data, features, pipe, result_dir, name1='confusion_matrix_baseline.png', name2='result_baseline.csv')

    print(f"Results saved to: {result_dir}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    result_dir = base_dir / "result"
    feature_path = result_dir / "feature_dataset.csv"  
    model_path = result_dir / "trained_decision_tree_model.joblib" 

    main_baseline_with_data(feature_path, result_dir, model_path=model_path)


# #########################################################################################################################
# #evaluating on unseen data, the trained model is saved in results directory
# #when run again on all data that it was trained on and tested on the results will not be perfect because function 
# #is not filtering good images and it havent seen test data, so it works only on dataset that has: good masks and one lesion per image

# def evaluate_model_on_unseen_all(
#     image_dir, mask_dir, metadata_path, model_path, result_dir
# ):
#     """
#     Evaluate trained model on unseen dataset using all available images in metadata.
#     Skips images if masks are missing. Extracts features and evaluates model.
#     """
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
#     extracted = feature_extraction(
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
#     features = ['border', 'asymmetry', 'mean_H', 'std_H', 'mean_S', 'std_S', 'mean_V', 'std_V', 'color_entropy', 'melanoma_colors']
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
#     plt.savefig(result_dir / "confusion_matrix_unseen.png", dpi=300)
#     plt.show()

#     results_df = pd.DataFrame({
#         "img_id": extracted["img_id"],
#         "patient_id": extracted["patient_id"],
#         "label": y_test,
#         "probability": y_prob
#     })
#     results_df.to_csv(result_dir / "results_unseen_test.csv", index=False)
#     print(f"Saved prediction results to {result_dir / 'results_unseen_test.csv'}")


# if __name__ == "__main__":
#     base_dir = Path(__file__).parent.resolve()

#     evaluate_model_on_unseen_all(
#         image_dir=base_dir / "data" / "skin_images" / "original",
#         mask_dir=base_dir / "data" / "lesion_masks",
#         metadata_path=base_dir / "dataset.csv",
#         model_path=base_dir / "result" / "trained_decision_tree_model.joblib",
#         result_dir=base_dir / "result"
#     )
