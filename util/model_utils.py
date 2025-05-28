import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score, 
    precision_score, f1_score, ConfusionMatrixDisplay

)
import matplotlib.pyplot as plt

'''
Objects for classifier.py
'''

#data oading and processing
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)
    data['cancer'] = data['cancer'].astype(int)
    
    X = data[['border', 'asymmetry', 'mean_H', 'std_H', 'mean_S', 'std_S', 'mean_V', 'std_V']]
    y = data['cancer']
    groups = data['patient_id']
    
    return data, X, y, groups

#performence metrics calculator.
def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    
    return metrics


#plots
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["Benign", "Cancer"],
        cmap="Oranges",
        values_format="d"
    )
    plt.title(title)
    plt.grid(False)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


#plots for cross validation results
def plot_cross_validation_results(summary_df, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.errorbar(summary_df['max_depth'], summary_df['mean_auc'], 
                yerr=summary_df['std_auc'], fmt='-o')
    plt.xlabel("Tree Depth")
    plt.ylabel("Mean AUC (±1 std)")
    plt.title("Decision Tree Cross-Validation for Depth Determination")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()