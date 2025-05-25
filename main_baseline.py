import sys
from os.path import join
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                           roc_auc_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def compute_metrics(y_true, y_pred, y_prob=None):
#Compute classification metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    return metrics

def main(features_path, label_path, save_path):
    """
    Train a model using the extracted dermoscopic features
    
    Args:
        features_path: Path to the extracted features CSV
        label_path: Path to the dataset CSV containing labels
        save_path: Path to save the results
    """
    
    
    #load extracted features
    print("Loading data...")
    try:
        features_df = pd.read_csv(features_path)
        labels_df = pd.read_csv(label_path)
        
    #error handling - file reafding
    except FileNotFoundError as e:
        print(f"Error: Could not find file: {e.filename}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: One of the input files is empty")
        return
    
    
    #merge features with labels
    print("Processing loaded data...")
    features_df['image_name'] = features_df['image_name'].str.replace('_mask', '')  # Remove _mask if present !!!
    merged_df = pd.merge(features_df, labels_df[['filename', 'label']], 
                        left_on='image_name', right_on='filename', how='inner')
    
    if merged_df.empty:
        print("Error: No matching records found after merging features with labels")
        return
    
    #features for training (select the ABC and or more / discard the rsst )
    feature_columns = ['pigment_network', 'blue_veil', 'vascular', 'globules', 'streaks',
                      'irregular_pigmentation', 'regression', 'asymmetry', 'compactness', 'convexity']
    
    #Sanity check for missing values
    missing_values = merged_df[feature_columns].isnull().sum()
    if missing_values.any():
        print("\nWarning !! : Missing values detected in features:")
        print(missing_values[missing_values > 0])
        print("\nRemoving rows with missing values...")
        merged_df = merged_df.dropna(subset=feature_columns)
        if merged_df.empty:
            print("Error: No records exsists !! ")
            return
    
    X = merged_df[feature_columns]
    y = merged_df['label']
      
    #Standardize features
    
    # Split the dataset
    
    #train classifier
    
    #pridictions 
    
    #metrics
    
    #save predictions and metrics
     
     