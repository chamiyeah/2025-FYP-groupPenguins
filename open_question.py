import pandas as pd
from util.classifier_final import prediction_evaluation, model_training
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

# ANSI color codes
YELLOW = '\033[93m'
PURPLE = '\033[95m'
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

def main_baseline_extra_feature(feature_path, result_dir):
    print(f"{YELLOW}Initializing baseline feature analysis...{RESET}")
    
    result_dir = Path(result_dir)
    feature_path = Path(feature_path)

    try:
        print(f"{YELLOW}Loading feature dataset from: {feature_path.name}{RESET}")
        data = pd.read_csv(feature_path)
        print(f"{GREEN}Successfully loaded feature dataset with {len(data)} samples{RESET}")
    except FileNotFoundError as e:
        print(f"{RED}Error loading extracted features CSV: {str(e)}{RESET}")
        return

    print(f"{YELLOW}Using extracted features from: {feature_path.name}{RESET}")

    features = ['border', 'asymmetry', 'texture', 'mean_H', 'std_H', 'mean_S', 'std_S', 'mean_V', 'std_V', 'color_entropy', 'melanoma_colors']#'hair_coverage' we exclude since baseline was better
    print(f"{PURPLE}Training model with {len(features)} features...{RESET}")

    pipe = model_training(data, features)
    print(f"{GREEN}Model training completed successfully{RESET}")

    print(f"{PURPLE}Evaluating model predictions...{RESET}")
    prediction_evaluation(data, features, pipe, result_dir, name1='confusion_matrix_open.png', name2='result_open.csv')

    print(f"{GREEN}Results saved successfully to: {result_dir}{RESET}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    result_dir = base_dir / "result"
    feature_path = result_dir / "feature_dataset.csv" #or feature_dataset_extended.csv
    model_path = result_dir / "trained_DT_extended.joblib"  

    main_baseline_extra_feature(feature_path, result_dir)

