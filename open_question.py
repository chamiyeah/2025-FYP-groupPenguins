import pandas as pd
from util.classifier_final import prediction_evaluation, model_training
from pathlib import Path
import joblib
import matplotlib.pyplot as plt



def main_baseline_extra_feature(feature_path, result_dir):

    result_dir = Path(result_dir)
    feature_path = Path(feature_path)

    try:
        data = pd.read_csv(feature_path)
    except FileNotFoundError as e:
        print(f"Error loading extracted features CSV: {str(e)}")
        return

    print(f"Using extracted features from: {feature_path.name}")

    features = ['border', 'asymmetry', 'texture', 'mean_H', 'std_H', 'mean_S', 'std_S', 'mean_V', 'std_V', 'color_entropy', 'melanoma_colors']#'hair_coverage' we exclude since baseline was better

    pipe = model_training(data, features)

    prediction_evaluation(data, features, pipe, result_dir, name1='confusion_matrix_open.png', name2='result_open.csv')

    print(f"Results saved to: {result_dir}")


if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    result_dir = base_dir / "result"
    feature_path = result_dir / "feature_dataset.csv" #or feature_dataset_extended.csv
    model_path = result_dir / "trained_DT_extended.joblib"  

    main_baseline_extra_feature(feature_path, result_dir)

