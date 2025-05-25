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
    """Compute classification metrics for multiclass classification"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
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
    merged_df = pd.merge(features_df, labels_df[['img_id', 'diagnostic']], 
                        left_on='image_name', right_on='img_id', how='inner')
    
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
    y = merged_df['diagnostic']
      
    #standardizing the features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    
    #splitting the dataste to train ¤ test 
    #from sklearn.model_selection import train_test_split < use this if the below giving erro r
    
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    #classifier training 
    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=1000, verbose=1)
    clf.fit(X_train, y_train)
    
    #make predictions - hooraaa !!
    print("\nMaking predictions...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]  # <<< Probability of positive class !!! 
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = compute_metrics(y_test, y_pred, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("\nResults:")
    print("-" * 50)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    #feature imoortance calc.
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': np.abs(clf.coef_[0])
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    
    #Saving results - has broken in to small blocks of code for eaiser tracking 
    
    print("\nSaving results...")
    test_indices = merged_df.index[y_test.index]
    result_df = merged_df.loc[test_indices, ['image_name', 'diagnostic']].copy()
    result_df['predicted_label'] = y_pred
    result_df['predicted_probability'] = y_prob
    result_df['correct_prediction'] = result_df['diagnostic'] == result_df['predicted_label']
    
    #add all metriks 
    for metric_name, metric_value in metrics.items():
        result_df[f'metric_{metric_name}'] = metric_value
    
    #savepredictions and metrics s 
    result_df.to_csv(save_path, index=False)
    print(f"Results saved to: {save_path}")
    
    #save feature importance to a separate file
    importance_path = save_path.replace('.csv', '_feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False)
    print(f"Feature importance saved to: {importance_path}")

if __name__ == "__main__":#
    features_path = "result/masked_out_images/extracted_features.csv"
    label_path = "dataset.csv"
    save_path = "result/result_main_baseline.csv"
    
    main(features_path, label_path, save_path)

